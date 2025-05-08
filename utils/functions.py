import re

def categorize_hl(row):
    if row <= 7200:
        return 'Low'
    elif 7200 < row <= 18000:
        return 'Medium'
    else:
        return 'High'

def float_or_false(value):
    try:
        if float(value):
            return True
    except ValueError:
        return False

def mod_col(df):
    df['modifications'] = df['N-terminus'] + ', ' + df['C-terminus']
    df['is_mod'] = 'True'

def mod_torf(df):
    for i in range(len(df)):
        if df.loc[i,'modifications'] != '':
            df.loc[i,'is_mod'] = 'True'
        else:
            df.loc[i,'is_mod'] = 'False'
'''
def mod_false(df, notna_func):
    for i in range(len(df)):
        if (
            df.loc[i, 'N-terminal_Modification'] != 'Free' or
            df.loc[i, 'C-terminal_Modification'] != 'Free' or
            df.loc[i, 'Other_Modification'] not in [None, 'None', '', nan]
        ):         
            df.loc[i,'is_mod'] = 'True'
            df.loc[i, 'modifications'] = f"{df.loc[i, 'N-terminal_Modification']}, {df.loc[i, 'C-terminal_Modification']}, {df.loc[i, 'Other_Modification']}"
        else:
            df.loc[i,'is_mod'] = 'False'
            df.loc[i,'modifications'] = ''
'''
def mod_false2(df):
    for i in range(len(df)):
        if df.loc[i,'nter'] != 'Free' or df.loc[i,'cter'] != 'Free':
            df.loc[i,'is_mod'] = 'True'
            df.loc[i,'modifications'] = f"{df.loc[i, 'nter']}, {df.loc[i, 'cter']}"
        else:
            df.loc[i,'is_mod'] = 'False'
            df.loc[i,'modifications'] = ''

#se agrega una nueva funcion mas optmizada para la funcion mod_false
def mod_false(df, notna_func):
    # Se crean las condiciones
    condition = (
        (df['N-terminal_Modification'] != 'Free') |
        (df['C-terminal_Modification'] != 'Free') |
        #preguntar por lo siguiente
        (df['Linear_Cyclic'] != 'Linear') |
        df['Other_Modification'].apply(notna_func)
    )
    
    # Asignar 'True' o 'False' en 'is_mod' basados en la condición
    df['is_mod'] = condition.map({True: 'True', False: 'False'})

    # Concatenar las modificaciones solo si existe alguna
    df['modifications'] = (
        df['N-terminal_Modification'].where(df['N-terminal_Modification'] != 'Free', '') + ', ' +
        df['C-terminal_Modification'].where(df['C-terminal_Modification'] != 'Free', '') + ', ' +
        df['Other_Modification'].where((df['Linear_Cyclic'] != 'Linear') & df['Other_Modification'].notna(), '') + ', ' +
        df['Linear_Cyclic'].where(df['Linear_Cyclic'] != 'Linear', '')  # Añadir 'Linear' solo si no es 'Linear'
    )
    
    # Eliminar la última coma y espacio en 'modifications' si la cadena está vacía
    df['modifications'] = df['modifications'].str.rstrip(', ')

    # Si no hay modificaciones, asignar un valor vacío
    df['modifications'] = df['modifications'].where(condition, '')


def handm_to_seconds(df, hl):
    df[hl] = df[hl].apply(
        lambda x: round(float(x.split(' ')[0]) * (3600 if 'hour' or 'hours' in x else 60), 5)
        ).astype(float)

def extract_data(text):
    return re.findall(r"'(.*?)'", text)

def canon_or_notcanon(sequence):
    alphabet = set("ACDEFGHIKLMNPQRSTVWY")
    sequence = sequence.strip()
    is_canon = True
    for res in set(sequence):
        if res not in alphabet:
            is_canon = False
    return is_canon

def show_duplicates(df, name, seq, charac_or_mod):
    df_dup = df[df.duplicated(subset=[seq, charac_or_mod], keep=False)].shape
    print(f"DataFrame: {name}, Duplicated Rows Shape: {df_dup}")
    df_nop_dup = df[~df.duplicated(subset=[seq, charac_or_mod], keep=False)].shape
    print(f"DataFrame: {name}, Unique Rows Shape: {df_nop_dup}")


def remove_duplicates(df, pd, seq, charac_or_mod, hl):
    # Separación de datos duplicados que tengan la misma secuencia y característica o modificación
    df_data_duplicated = df[df.duplicated(subset=[seq, charac_or_mod], keep=False)].sort_values(by=seq)
    
    # Separación de datos duplicados que tengan la misma secuencia, característica o modificación y half_life_seconds
    df_data_duplicated_2 = df[df.duplicated(subset=[seq, charac_or_mod, hl], keep=False)].sort_values(by=seq)
    
    # Separación de datos no duplicados
    df_not_duplicated = df[~df.duplicated(subset=[seq, charac_or_mod], keep=False)].sort_values(by=seq)
    
    # Concatenar los datos no duplicados con los duplicados pero que se mantienen
    df_not_duplicated = pd.concat([df_not_duplicated, df_data_duplicated_2])

    # Eliminar valores nulos y duplicados sobrantes
    df_not_duplicated.dropna(subset=[seq, hl], inplace=True)
    df_not_duplicated.drop_duplicates(subset=[seq, charac_or_mod, hl], keep='first', inplace=True)
    
    # Ordenar el DataFrame
    df_not_duplicated.sort_index(inplace=True)

    return df_not_duplicated
