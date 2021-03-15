import pandas as pd


def dtype_map(dtype):
    """
    Map Python data types to Avro types
    
    param: dtype: Python data type
    return: corresponding Avro type
    """

    if dtype=='int64' or dtype=='int32':
        return 'int'
    elif dtype=='float64' or dtype=='float32':
        return 'float'
    elif dtype=='O':
        return 'string'
    elif dtype=='bool':
        return 'boolean'
    else:
        return None


def dataClass_map(data_type):
    """
    Map Avro types to dataClass (numerocal vs categorical)

    param: data_type: Avro data type
    return: numerical vs categorical
    """

    if data_type=='string' or data_type=='boolean':
        return 'categorical'
    elif data_type=='float' or data_type=='int':
        return 'numerical'
    else:
        return None


def role_map(field_name):
    """
    Map DataFrame fields to role

    param: field_name: DF field
    return: 'label', 'score', 'predictor' or 'non-predictor'
    """

    field_name = field_name.lower()

    if field_name=='label' or field_name=='score':
        return field_name
    elif field_name=='id':
        return 'non-predictor'
    else:
        return 'predictor'


# Expanded Schema metadata
metadata_values = {
    'type': ['int', 'float', 'string', 'boolean', None],
    'dataClass': ['numerical', 'categorical', None],
    'role': ['label', 'score', 'predictor', 'non-predictor'],
    'protectedClass': [True, False, 0, 1],
    'driftCandidate': [True, False, 0, 1]
}


def infer_schema(
    input_format=None, 
    dataframe=None, 
    filename=None
    ):

    """
    A function to infer an expanded schema from input df or file
    
    param: input_format: 'file' or 'df'
    param: dataframe: input dataframe
    param: fielname: input filename

    return: Schema containing metadata for all input data fields
    """

    if input_format=='file':
        dataframe = pd.read_json(filename, orient='records', lines=True)

    # Schema fields
    fields=[]
    dtypes=dict(dataframe.dtypes) # Pandas dtypes

    for field in dataframe.columns.values:
        # Map Pandas dtypes to AVRO types
        field_type = dtype_map(dtypes[field])

        fields.append(
            {
                'name': field,
                'type': field_type,
                'dataClass': dataClass_map(field_type),
                'role': role_map(field),
                'protectedClass': False,
                'driftCandidate': True,
                'specialValues': []
            }
        )

    schema_df = pd.DataFrame(fields)
    schema_df.set_index('name', inplace=True)

    return schema_df


def validate_schema(dataframe):
    
    is_validated = True

    for field in dataframe.index:
        for metadata in [
            'type', 
            'dataClass', 
            'role', 
            'protectedClass', 
            'driftCandidate']:

            if dataframe.loc[field, metadata] not in metadata_values[metadata]:
                print('{} = {} not in {}'.format(
                    metadata, dataframe.loc[field, metadata], metadata_values[metadata]
                    )
                )
                is_validated = False

    return is_validated


def set_detector_parameters(schema_df):
    """
    A function to set defaults for detectors

    param: schema_df: expanded schema of input data

    return: map of detectore parameters
    """

    categorical_columns = []
    numeric_columns = []
    score_column = []
    label_column = []

    if validate_schema(schema_df):

        for field in schema_df.index.values:

            if schema_df.loc[field, 'driftCandidate']==True:

                if schema_df.loc[field, 'dataClass']=='categorical':
                    categorical_columns.append(field)
                elif schema_df.loc[field, 'dataClass']=='numerical':
                    numeric_columns.append(field)
            
            if schema_df.loc[field, 'role']=='score':
                score_column.append(field)
            elif schema_df.loc[field, 'role']=='label':
                label_column.append(field)
    else:
        print('\nSchema did not pass validation. Setting parameters to defaults.')
    
    return {
        'categorical_columns': categorical_columns, 
        'numerical_columns': numeric_columns, 
        'score_column': score_column, 
        'label_column': label_column
        }
