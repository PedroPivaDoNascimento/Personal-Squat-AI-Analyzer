"""
Serviço de limpeza autônoma de planilhas - Camada de Serviço (Service Layer)

Responsabilidade única: varrer o diretório `planilhas/` (e todas as subpastas
geradas por `SetFolders`, ex.: `frontal/`, `sagital/`), identificar a planilha
`.xlsx` mais antiga pelo timestamp de modificação (`mtime`, com fallback para
`ctime`) e removê-la fisicamente.

Resiliência garantida (docs/RULES.md / Fase 3 do plano de tarefas):
- Pasta inexistente ou vazia -> rotina NÃO quebra, apenas registra aviso;
- Erros de I/O (permissão, arquivo sumido, disco) -> capturados e logados;
- Logging estruturado registrando nome do arquivo, caminho completo e timestamp.
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Extensão-alvo da limpeza (somente relatórios Excel são elegíveis para exclusão)
XLSX_EXTENSION = ".xlsx"


@dataclass(frozen=True)
class CleanupResult:
    """Resultado estruturado de um ciclo de limpeza."""

    deleted: bool
    file_path: str | None = None
    file_name: str | None = None
    mtime_iso: str | None = None
    reason: str = ""


class SheetCleanupService:
    """
    Serviço responsável pela remoção periódica da planilha `.xlsx` mais antiga
    do repositório de relatórios (`planilhas/`).

    Framework-agnostic: pode ser invocado pelo Django Management Command
    `cleanup_oldest_sheet`, pelo agendador APScheduler ou por testes unitários.
    """

    def __init__(self, sheets_root: str | os.PathLike | None = None):
        """
        Args:
            sheets_root: Raiz do diretório de planilhas. Quando `None`, resolve
                a partir da configuração central `PLANILHAS_ROOT_DIR` do Django
                (settings.py); fora de contexto Django usa a raiz do projeto
                (`<BASE_DIR>/planilhas`) como fallback determinístico.
        """
        if sheets_root is not None:
            self.sheets_root = Path(sheets_root).resolve()
        else:
            self.sheets_root = self._resolve_default_root()

    @staticmethod
    def _resolve_default_root() -> Path:
        """Resolve a raiz padrão das planilhas com ou sem contexto Django."""
        try:
            from django.conf import settings

            root = getattr(settings, "PLANILHAS_ROOT_DIR", None)
            if root is not None:
                return Path(root).resolve()
        except Exception:  # pragma: no cover - ambiente sem Django configurado
            pass
        project_root = Path(__file__).resolve().parent.parent.parent
        return (project_root / "planilhas").resolve()

    # ------------------------------------------------------------------ #
    # Varredura e seleção                                                #
    # ------------------------------------------------------------------ #
    def find_all_xlsx(self) -> list[Path]:
        """
        Varre recursivamente `planilhas/` (subpastas `frontal/`, `sagital/` e
        aninhadas como `dados_pe/dados brutos/`) filtrando apenas arquivos
        `.xlsx`. Pastas inexistentes ou erros de permissão durante o walk são
        tratados graciosamente sem interromper a rotina.

        Returns:
            Lista de caminhos absolutos dos arquivos `.xlsx` encontrados.
        """
        if not self.sheets_root.exists():
            logger.warning(
                "cleanup.sheets_root_missing root=%s "
                "message='Diretorio de planilhas nao existe; nada a limpar.'",
                self.sheets_root,
            )
            return []
        if not self.sheets_root.is_dir():
            logger.error(
                "cleanup.sheets_root_invalid root=%s "
                "message='Caminho de planilhas existe mas nao e um diretorio.'",
                self.sheets_root,
            )
            return []

        found: list[Path] = []
        # onerror: registra e continua — uma subpasta ilegivel nao derruba o ciclo
        for dirpath, _dirnames, filenames in os.walk(self.sheets_root, onerror=self._log_walk_error):
            for filename in filenames:
                if filename.lower().endswith(XLSX_EXTENSION):
                    candidate = Path(dirpath) / filename
                    # Ignora temporários ocultos do Excel (~$arquivo.xlsx)
                    if filename.startswith("~$"):
                        continue
                    found.append(candidate)
        return found

    @staticmethod
    def _file_timestamp(path: Path) -> float | None:
        """
        Algoritmo de comparação de timestamp: retorna o `mtime` do arquivo e,
        em caso de falha de I/O nesse arquivo específico, faz fallback para
        `ctime`. Retorna `None` se o arquivo não puder ser estatisticado.
        """
        try:
            stat_result = path.stat()
            mtime = stat_result.st_mtime
            # Em alguns sistemas de arquivos o mtime pode ser 0/irreal; usa ctime como fallback
            if mtime and mtime > 0:
                return mtime
            return stat_result.st_ctime
        except OSError as exc:
            logger.warning(
                "cleanup.stat_failed file=%s error='%s' message='Arquivo ignorado na selecao.'",
                path, exc,
            )
            return None

    def find_oldest_xlsx(self) -> tuple[Path, float] | None:
        """
        Percorre os `.xlsx` candidatos e seleciona unicamente o mais antigo
        pelo menor timestamp (`mtime`/`ctime`).

        Returns:
            Tupla (caminho, timestamp) do arquivo mais antigo, ou `None` se
            nenhuma planilha elegível existir.
        """
        candidates = self.find_all_xlsx()
        if not candidates:
            logger.info(
                "cleanup.no_sheets root=%s message='Nenhuma planilha .xlsx encontrada; ciclo encerrado sem remocao.'",
                self.sheets_root,
            )
            return None

        oldest_path: Path | None = None
        oldest_ts: float | None = None
        for candidate in candidates:
            ts = self._file_timestamp(candidate)
            if ts is None:
                continue
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts
                oldest_path = candidate

        if oldest_path is None or oldest_ts is None:
            logger.warning(
                "cleanup.no_statable_files root=%s message='Planilhas encontradas mas nenhum timestamp disponivel.'",
                self.sheets_root,
            )
            return None
        return oldest_path, oldest_ts

    # ------------------------------------------------------------------ #
    # Remoção                                                            #
    # ------------------------------------------------------------------ #
    def delete_oldest_sheet(self) -> CleanupResult:
        """
        Executa um ciclo completo de limpeza: varre, seleciona a planilha
        `.xlsx` mais antiga e a remove fisicamente de forma permanente.

        Returns:
            CleanupResult indicando se houve remoção e os metadados do arquivo.
        """
        oldest = self.find_oldest_xlsx()
        if oldest is None:
            return CleanupResult(deleted=False, reason="no_xlsx_files")

        file_path, timestamp = oldest
        mtime_iso = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

        try:
            file_path.unlink()  # remoção física permanente
        except FileNotFoundError:
            # Corrida benigna: outro processo já removeu o arquivo.
            logger.warning(
                "cleanup.delete_race file=%s path=%s "
                "message='Arquivo ja removido por outro processo; ciclo concluido sem erro.'",
                file_path.name, file_path,
            )
            return CleanupResult(
                deleted=False, file_path=str(file_path),
                file_name=file_path.name, mtime_iso=mtime_iso,
                reason="file_already_removed",
            )
        except PermissionError as exc:
            logger.error(
                "cleanup.delete_permission_denied file=%s path=%s error='%s' "
                "message='Sem permissao para excluir; verifique volume/mount do Docker.'",
                file_path.name, file_path, exc,
            )
            return CleanupResult(
                deleted=False, file_path=str(file_path),
                file_name=file_path.name, mtime_iso=mtime_iso,
                reason="permission_denied",
            )
        except OSError as exc:
            logger.exception(
                "cleanup.delete_io_error file=%s path=%s error='%s'",
                file_path.name, file_path, exc,
            )
            return CleanupResult(
                deleted=False, file_path=str(file_path),
                file_name=file_path.name, mtime_iso=mtime_iso,
                reason="io_error",
            )

        deletion_ts = datetime.now(timezone.utc).isoformat()
        logger.info(
            "cleanup.sheet_deleted file=%s path=%s oldest_mtime=%s deleted_at=%s "
            "message='Remocao periodica autonoma concluida com sucesso.'",
            file_path.name, file_path, mtime_iso, deletion_ts,
        )
        return CleanupResult(
            deleted=True, file_path=str(file_path),
            file_name=file_path.name, mtime_iso=mtime_iso,
            reason="deleted",
        )

    # ------------------------------------------------------------------ #
    # Utilitários                                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _log_walk_error(error: OSError) -> None:
        """Handler de erro do os.walk: registra e permite continuar a varredura."""
        logger.warning(
            "cleanup.walk_error path='%s' error='%s' message='Subpasta ignorada na varredura.'",
            getattr(error, "filename", "<desconhecido>"), error,
        )
