"""
Alias de compatibilidade (Nota de Migração da task (3)NEW-DATA_ANALYTICS-SHEETS.md).

A implementação do escritor de dados de pé no plano frontal foi migrada para
`src/classes/excel/segments_writer/frontal_foot_data_excel_writer.py`
(classe `FrontalFootDataExcelWriter`, herdeira de `SegmentsWriter`).

Este módulo mantém o caminho de importação legado
`classes.excel.foot_data_excel_writer.FootDataExcelWriter` funcionando durante
a transição, sem duplicar lógica. Novos códigos devem importar diretamente:

    from classes.excel.segments_writer.frontal_foot_data_excel_writer import FrontalFootDataExcelWriter
"""
from classes.excel.segments_writer.frontal_foot_data_excel_writer import (
    FrontalFootDataExcelWriter,
)

# Alias de compatibilidade para a classe legada (assinatura e comportamento
# idênticos ao antigo FootDataExcelWriter; apenas renomeada/migrada).
FootDataExcelWriter = FrontalFootDataExcelWriter

__all__ = ['FootDataExcelWriter', 'FrontalFootDataExcelWriter']
