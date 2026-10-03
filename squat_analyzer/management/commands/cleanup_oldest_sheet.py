"""
Django Management Command: cleanup_oldest_sheet

Executa a limpeza autônoma do diretório `planilhas/`, removendo fisicamente a
planilha `.xlsx` mais antiga (critério: menor `mtime`, fallback `ctime`).

Uso unitário (ex.: via crontab horária):
    python manage.py cleanup_oldest_sheet

Uso como agendador em segundo plano 24/7 — disparo exato a cada 1 hora,
independente de acessos de usuários ao site:
    python manage.py cleanup_oldest_sheet --loop

A lógica de varredura/seleção/remoção reside exclusivamente na camada de
serviço (`squat_analyzer.services.cleanup_service.SheetCleanupService`),
seguindo o padrão Service Layer do projeto (docs/RULES.md, Seção 2.1).
"""
import logging
import time
from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError

from squat_analyzer.services.cleanup_service import SheetCleanupService

logger = logging.getLogger(__name__)

# Intervalo exato da rotina periódica autônoma: 1 hora
CLEANUP_INTERVAL_SECONDS = 60 * 60


class Command(BaseCommand):
    help = (
        "Remove a planilha .xlsx mais antiga do diretorio planilhas/ "
        "(execucao unica ou agendada a cada 1 hora com --loop)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--loop",
            action="store_true",
            dest="loop",
            help=(
                "Mantem o processo vivo executando a limpeza em segundo plano "
                "a cada 1 hora (background job 24/7 para uso no container worker)."
            ),
        )
        parser.add_argument(
            "--interval-seconds",
            type=int,
            default=CLEANUP_INTERVAL_SECONDS,
            dest="interval_seconds",
            help="Intervalo entre ciclos em segundos (padrao: 3600 = 1 hora).",
        )
        parser.add_argument(
            "--sheets-root",
            type=str,
            default=None,
            dest="sheets_root",
            help="Override do diretorio raiz das planilhas (default: <BASE_DIR>/planilhas).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            dest="dry_run",
            help="Identifica e loga a planilha mais antiga sem remove-la.",
        )

    def handle(self, *args, **options):
        service = SheetCleanupService(sheets_root=options.get("sheets_root"))
        loop = options["loop"]
        interval = max(int(options["interval_seconds"]), 1)

        if loop:
            self._run_scheduler(service, interval, dry_run=options["dry_run"])
        else:
            self._run_once(service, dry_run=options["dry_run"])

    # ------------------------------------------------------------------ #
    # Execução unitária                                                  #
    # ------------------------------------------------------------------ #
    def _run_once(self, service: SheetCleanupService, dry_run: bool = False,
                  swallow_errors: bool = False) -> None:
        """
        Executa um único ciclo de limpeza com tratamento completo de exceções.

        Args:
            service: Instância do SheetCleanupService (camada de serviço).
            dry_run: Quando True, apenas identifica a planilha mais antiga.
            swallow_errors: Quando True (modo agendador), erros inesperados são
                logados sem interromper os próximos ciclos horários.
        """
        try:
            if dry_run:
                oldest = service.find_oldest_xlsx()
                if oldest is None:
                    self.stdout.write(self.style.WARNING(
                        "[DRY-RUN] Nenhuma planilha .xlsx encontrada em "
                        f"{service.sheets_root}. Nada seria removido."
                    ))
                    return
                path, ts = oldest
                self.stdout.write(self.style.NOTICE(
                    f"[DRY-RUN] Planilha mais antiga: {path} (mtime={ts})"
                ))
                return

            result = service.delete_oldest_sheet()
        except Exception as exc:  # nunca derruba o agendador/cron por erro inesperado
            logger.exception("cleanup.unexpected_error error='%s'", exc)
            if swallow_errors:
                logger.error(
                    "cleanup.cycle_failed error='%s' message='Ciclo falhou; agendador seguira no proximo disparo.'",
                    exc,
                )
                return
            raise CommandError(f"Falha inesperada na limpeza de planilhas: {exc}") from exc

        if result.deleted:
            self.stdout.write(self.style.SUCCESS(
                f"Planilha removida: {result.file_name} | caminho: {result.file_path} "
                f"| mtime original: {result.mtime_iso}"
            ))
        elif result.reason == "no_xlsx_files":
            self.stdout.write(self.style.WARNING(
                f"Diretorio {service.sheets_root} sem planilhas .xlsx - "
                "ciclo concluido graciosamente sem remocao."
            ))
        else:
            self.stdout.write(self.style.WARNING(
                f"Nenhuma remocao efetuada (motivo: {result.reason}) - "
                f"arquivo analisado: {result.file_path}"
            ))

    # ------------------------------------------------------------------ #
    # Agendador em segundo plano (APScheduler - Background Job 24/7)     #
    # ------------------------------------------------------------------ #
    def _run_scheduler(self, service: SheetCleanupService, interval: int, dry_run: bool = False) -> None:
        """
        Background job 24/7 com APScheduler (BlockingScheduler): executa o
        primeiro ciclo imediatamente e repete a cada `interval` segundos
        (padrão: exatos 1 hora), sem depender de acessos de usuários ao site.

        Se o APScheduler não estiver disponível no ambiente, há fallback
        determinístico para um loop resiliente baseado em `time.sleep`.
        Qualquer exceção dentro de um ciclo é capturada e logada — a rotina
        horária jamais é interrompida.
        """
        logger.info(
            "cleanup.scheduler_start interval_seconds=%s root=%s "
            "message='Agendador de limpeza autonoma iniciado (disparo a cada 1 hora).'",
            interval, service.sheets_root,
        )
        self.stdout.write(self.style.NOTICE(
            f"Agendador de limpeza iniciado: disparo imediato e depois "
            f"a cada {timedelta(seconds=interval)}."
        ))

        cycle = lambda: self._run_once(service, dry_run=dry_run, swallow_errors=True)  # noqa: E731

        try:
            from apscheduler.schedulers.blocking import BlockingScheduler

            scheduler = BlockingScheduler(timezone="UTC")
            scheduler.add_job(
                cycle,
                trigger="interval",
                seconds=interval,
                id="cleanup_oldest_sheet_hourly",
                name="Limpeza autonoma da planilha .xlsx mais antiga",
                max_instances=1,
                coalesce=True,
                misfire_grace_time=interval,
            )
            # Execução imediata do primeiro ciclo; após isso, a cada 1 hora.
            cycle()
            try:
                scheduler.start()
            except (KeyboardInterrupt, SystemExit):
                logger.info("cleanup.scheduler_stopped message='Encerramento manual do agendador.'")
            return
        except ImportError:
            logger.warning(
                "cleanup.apscheduler_missing message='APScheduler indisponivel; usando fallback time.sleep.'"
            )

        # Fallback resiliente sem dependências externas
        while True:
            cycle()
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("cleanup.scheduler_stopped message='Encerramento manual do agendador.'")
                return
