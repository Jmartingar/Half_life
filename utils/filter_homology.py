import os
import pandas as pd
from Bio import SeqIO
import warnings
warnings.filterwarnings("ignore")

class HomologyReduction:
    def parse_fasta_file(self, file_path):
        data = list(SeqIO.parse(file_path, format="fasta"))
        data = [[str(record.seq)] for record in data]
        df = pd.DataFrame(data, columns=["sequence"])
        return df    
        
    def pandas_to_fasta(self, df, output_name):
        doc_open = open(output_name, "w")

        for index in df.index:
            id_value = df["id"][index]
            sequence = df["sequence"][index]

            doc_open.write(f">seq{id_value}\n")
            doc_open.write(f"{sequence}\n")
        doc_open.close()


    def convert_result_to_csv(self, fasta_file, data_type, path_export):
        csv_export = f"../../data/{path_export}/csv_files"
        os.makedirs(csv_export, exist_ok=True)

        df = self.parse_fasta_file(fasta_file)

        df.to_csv(f"{csv_export}/{data_type}_data_filter.csv", index=False)
        

    def process(self, df=None, data_type=None, value_filter=None, path_export=None, len=None):
        fasta_export = f"../../data/{path_export}/fasta_files"
        os.makedirs(fasta_export, exist_ok=True)

        df["id"] = df.index + 1
        self.pandas_to_fasta(df, f"{fasta_export}/{data_type}_data.fasta")

        command = f"cd-hit -i {fasta_export}/{data_type}_data.fasta -o {fasta_export}/{data_type}_data_filter.fasta -c {value_filter} -l {len} -M 1000"  # l corresponde al largo de descarte de secuencias (largo minimo)
        os.system(command)                                                                                                                                         # M more memory that default

        self.convert_result_to_csv(f"{fasta_export}/{data_type}_data_filter.fasta", data_type, path_export)