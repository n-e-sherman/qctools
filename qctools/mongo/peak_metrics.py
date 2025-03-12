from mongoengine import EmbeddedDocument, FloatField, StringField

class PeakMetric(EmbeddedDocument):

    time = FloatField()
    peak = FloatField()
    variant = StringField() # used to check if this metric has already been computed
    comment = StringField()
    device = StringField(default='cpu')

    meta = {'allow_inheritance': True}

from mongoengine import EmbeddedDocument, FloatField, StringField, DictField

class StatevectorPeakMetric(PeakMetric):

    method = StringField(default="SV")
    time_simulation = FloatField()
    time_sample = FloatField()
    probabilities = DictField()
    
from mongoengine import EmbeddedDocument, FloatField, StringField, DictField, IntField

class MPSPeakMetric(PeakMetric):

    method = StringField(default="MPS")
    chi = IntField()
    time_simulation = FloatField()
    time_sample = FloatField()
    probabilities = DictField()