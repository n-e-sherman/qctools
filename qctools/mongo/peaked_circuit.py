import pandas as pd
import numpy as np
from mongoengine import Document, StringField, IntField, FloatField, DictField, FileField, EmbeddedDocumentField, EmbeddedDocumentListField, ListField

from .peak_metrics import PeakMetric

class PeakedCircuit(Document):

    
    num_qubits = IntField(required=True)
    target = StringField()
    num_gates = IntField()
    depth = IntField()
    num_two_qubit = IntField()
    peak = FloatField()

    mlflow_run_id = StringField() # sort of want this to be required
    qasm = FileField()
    comment = StringField()

    peak_metrics = EmbeddedDocumentListField(PeakMetric)
    best_peak_metric = EmbeddedDocumentField(PeakMetric)

    # 🔹 Override delete() to always remove the QASM file
    def delete(self, *args, **kwargs):
        if self.qasm and self.qasm.grid_id:  # Check if a QASM file exists
            self.qasm.delete()  # Delete the actual file in GridFS
        super().delete(*args, **kwargs)  # Now delete the document itself

    def add_peak_metric(self, metric):

        self.peak_metrics.append(metric)
        self.update_best_peak_metric()

    def update_best_peak_metric(self):

        times = [metric.time for metric in self.peak_metrics if metric.peak_found==True]
        if len(times) > 0:
            i_best = np.argmin(times)
            self.best_peak_metric = self.peak_metrics[i_best]

    def to_df(self, skips = []):

        data_dict = self.to_mongo().to_dict()
        res_dict = {}
        for k,v in data_dict.items():

            if k in skips:
                continue

            elif isinstance(v, list):
                continue

            elif isinstance(v, dict):
                if k == 'best_peak_metric':
                    new_v = {f'best_{_k}': _v for _k, _v in v.items()}
                    res_dict.update(new_v)
                    continue
                else:
                    continue
            res_dict[k] = v

        try:
            return pd.DataFrame(res_dict)
        except:
            return pd.DataFrame([res_dict])


