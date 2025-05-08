import pandas as pd
import numpy as np
import qiskit
from qiskit import qasm2, transpile
from mongoengine import Document, StringField, IntField, FloatField, DictField, FileField, EmbeddedDocumentField, EmbeddedDocumentListField, ListField

from .peak_metrics import PeakMetric
from .bq_metrics import BQJobMetric
# from qctools.mongo import PeakMetric

class PeakedCircuit(Document):

    num_qubits = IntField(required=True)
    target = StringField(required=True)
    mlflow_run_id = StringField()
    depth = IntField()
    comment = StringField()

    # derived in  __init__
    peak = FloatField()
    num_two_qubit = IntField()
    num_gates = IntField()
    qasm = FileField()
    
    # added through specific function
    bq_metrics = EmbeddedDocumentListField(BQJobMetric)
    best_bq_metric = EmbeddedDocumentField(BQJobMetric)
    
    peak_metrics = EmbeddedDocumentListField(PeakMetric)
    best_peak_metric = EmbeddedDocumentField(PeakMetric)
    

    def __init__(self, qc, *args, basis_gates=['u3', 'cz'], optimization_level=1, **kwargs):

        qct = transpile(qc, basis_gates=basis_gates, optimization_level=optimization_level)

        kwargs['num_qubits'] = kwargs.get('num_qubits', qct.num_qubits)
        kwargs['num_two_qubit'] = kwargs.get('num_two_qubit', qct.count_ops().get('cz', 0))
        kwargs['num_gates'] = kwargs.get('num_gates', int(np.sum(list(qct.count_ops().values()))))
        
        super().__init__(*args, **kwargs)

        if qct:
            qasm_str = qasm2.dumps(qct)
            self.qasm.put(qasm_str.encode("utf-8"), content_type="text/plain")
            
    # 🔹 Override delete() to always remove the QASM file
    def delete(self, *args, **kwargs):
        if self.qasm and self.qasm.grid_id:  # Check if a QASM file exists
            self.qasm.delete()  # Delete the actual file in GridFS
        super().delete(*args, **kwargs)  # Now delete the document itself

    def add_bq_metric(self, metric):
        self.bq_metrics.append(metric)
        self.update_best_bq_metric()

    def add_peak_metric(self, metric):

        self.peak_metrics.append(metric)
        self.update_best_peak_metric()

    def get_qasm(self):

        self.qasm.seek(0)
        return self.qasm.read().decode("utf-8")
    
    def get_qc(self):

        qasm = self.get_qasm()
        return qiskit.QuantumCircuit.from_qasm_str(qasm)

    def update_best_peak_metric(self):

        self.best_peak_metric = None # Is this allowed?
        times = [metric.time for metric in self.peak_metrics if metric.peak_found==True]
        if len(times) > 0:
            min_time = np.min(times)
            for metric in self.peak_metrics:
                if metric.time == min_time:
                    self.best_peak_metric = metric

    def update_best_bq_metric(self):

        self.best_bq_metric = None # Is this allowed?
        times = [metric.time for metric in self.bq_metrics if metric.target_found==True]
        if len(times) > 0:
            min_time = np.min(times)
            for metric in self.bq_metrics:
                if metric.time == min_time:
                    self.best_bq_metric = metric

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
                    new_v = {f'best_peak_{_k}': _v for _k, _v in v.items() if not _k == 'probabilities'}
                    res_dict.update(new_v)
                    continue
                elif k == 'best_bq_metric':
                    new_v = {f'best_{_k}': _v for _k, _v in v.items() if not _k == 'probabilities'}
                    res_dict.update(new_v)
                    continue
                    
                else:
                    continue
            res_dict[k] = v

        try:
            return pd.DataFrame(res_dict)
        except:
            return pd.DataFrame([res_dict])
