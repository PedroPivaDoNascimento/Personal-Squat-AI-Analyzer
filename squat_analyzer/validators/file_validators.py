import os
from django.core.exceptions import ValidationError
from django.template.defaultfilters import filesizeformat
from django.utils.translation import gettext_lazy as _


class MP4VideoValidator:
    """
    Validador personalizado para garantir que apenas arquivos .mp4 sejam aceitos.

    Este validador verifica tanto a extensão do arquivo quanto o tipo MIME
    para garantir máxima segurança contra uploads maliciosos.

    Uso em Models:
        video = models.FileField(
            upload_to='videos/',
            validators=[MP4VideoValidator()]
        )

    Uso em Forms:
        video = forms.FileField(
            validators=[MP4VideoValidator()]
        )
    """

    # Extensões permitidas (apenas .mp4)
    ALLOWED_EXTENSIONS = ['.mp4']

    # Tipos MIME permitidos para MP4
    ALLOWED_MIME_TYPES = [
        'video/mp4',
        'video/x-m4v',
    ]

    # Tamanho máximo padrão (100MB)
    DEFAULT_MAX_SIZE = 102428800  # 100MB em bytes

    def __init__(self, max_size=None):
        """
        Inicializa o validador.

        Args:
            max_size: Tamanho máximo do arquivo em bytes. Se None, usa o DEFAULT_MAX_SIZE.
        """
        self.max_size = max_size or self.DEFAULT_MAX_SIZE

    def __call__(self, file):
        """
        Executa a validação do arquivo.

        Args:
            file: Objeto UploadedFile do Django

        Raises:
            ValidationError: Se o arquivo não for válido
        """
        # Valida a extensão do arquivo
        self._validate_extension(file)

        # Valida o tipo MIME
        self._validate_mime_type(file)

        # Valida o tamanho do arquivo
        self._validate_file_size(file)

    def _validate_extension(self, file):
        """
        Valida a extensão do arquivo.

        Args:
            file: Objeto UploadedFile do Django

        Raises:
            ValidationError: Se a extensão não for .mp4
        """
        ext = os.path.splitext(file.name)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValidationError(
                _('Apenas arquivos MP4 são permitidos. Você enviou: %(filename)s'),
                params={'filename': file.name},
                code='invalid_extension'
            )

    def _validate_mime_type(self, file):
        """
        Valida o tipo MIME do arquivo.

        Esta é uma camada adicional de segurança, pois a extensão
        do arquivo pode ser facilmente falsificada.

        Args:
            file: Objeto UploadedFile do Django

        Raises:
            ValidationError: Se o tipo MIME não for permitido
        """
        # Tenta obter o content_type do arquivo
        if hasattr(file, 'content_type') and file.content_type:
            mime_type = file.content_type
            if mime_type not in self.ALLOWED_MIME_TYPES:
                raise ValidationError(
                    _('Tipo de arquivo não suportado: %(content_type)s. Apenas MP4 é permitido.'),
                    params={'content_type': mime_type},
                    code='invalid_mime_type'
                )

    def _validate_file_size(self, file):
        """
        Valida o tamanho do arquivo.

        Args:
            file: Objeto UploadedFile do Django

        Raises:
            ValidationError: Se o arquivo for muito grande
        """
        if file.size > self.max_size:
            raise ValidationError(
                _('O arquivo é muito grande (%(size)s). O tamanho máximo permitido é %(max_size)s.'),
                params={
                    'size': filesizeformat(file.size),
                    'max_size': filesizeformat(self.max_size)
                },
                code='file_too_large'
            )


def validate_mp4_video(value):
    """
    Função validadora simples para validar arquivos MP4.

    Esta é uma função convenience que pode ser usada diretamente
    como validador em campos de formulário ou modelo.

    Uso:
        from squat_analyzer.validators import validate_mp4_video

        class VideoForm(forms.Form):
            video = forms.FileField(validators=[validate_mp4_video])

    Args:
        value: Objeto UploadedFile do Django

    Raises:
        ValidationError: Se o arquivo não for um MP4 válido
    """
    validator = MP4VideoValidator()
    validator(value)