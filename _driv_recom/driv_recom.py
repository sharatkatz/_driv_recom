"""Main module."""
#!/usr/bin/python
# -*- coding: UTF-8 -*-
def get_standard_product():
    """[Fetch data from Table]

    Returns:
        [dataframe]: [data]
    """
    import pyodbc
    import pandas as pd
    conn = pyodbc.connect(
        r'DRIVER={ODBC Driver 17 for SQL Server};'
    )
    print("\n-------Fetching data-------")
    query = \
        """
    SELECT *
        FROM Table
    """
    raw = pd.read_sql(query, conn)
    print()
    # print(raw.info())
    return raw

def get__LOGS():
    """[return _LOGS data]

    Returns:
        [dataframe]: [SQL query for
        return _LOGS data]
    """
    import pyodbc
    import pandas as pd
    conn = pyodbc.connect(
        r'DRIVER={ODBC Driver 17 for SQL Server};'
    )
    print("\n-------Fetching _LOGS data for specified duration-------")
    query = \
        """
    SELECT _key, _time_, _ID, _BRAND, LOB
    FROM Table with (nolock)
    WHERE _key IN (
      SELECT MAX(_key) FROM Table with (nolock)
          WHERE SERVER_NAME = Server_Name
          AND _ID is not null
          AND symptom is not null
          GROUP BY _ID, symptom
        )
    ORDER BY _key, _time_
    """
    raw = pd.read_sql(query, conn)
    print("\n-------Printing _LOGS info-------")
    print(raw.info())
    return raw


def get__LOGS():
    """[return _LOGS data]

    Returns:
        [dataframe]: [SQL query for
        return _LOGS data]
    """
    print("\n-------Fetching _LOGS data-------")
    raw_logs = pd.read_csv(
        "../Data/_LOGS_PARSED_recent.csv")
    print(raw_logs.info())
    return raw_logs


def is_LOB(v):
    """[summary]

    Args:
        v ([String]): [Input string]

    Returns:
        [Boolean]: [check if input string is LOB like]
    """
    v = str(v)
    if v.upper() in [LOBs]:
        return True
    else:
        return False


def is_dict_list(v):
    """[summary]

    Args:
        v ([variable]): [python variable]

    Returns:
        [boolean]: [is specified variable a dictionary or list]
    """
    return isinstance(v, (dict, list))


def is_dict(v):
    """[summary]

    Args:
        v ([variable]): [python variable]

    Returns:
        [boolean]: [is specified variable a dictionary]
    """
    return isinstance(v, dict)


def is_list(v):
    """[summary]

    Args:
        v ([variable]): [python variable]

    Returns:
        [boolean]: [is specified variable a list]
    """
    return isinstance(v, list)


def Brand_Object(idx):
    """
    return dict value at certain level
    """
    return dictt[idx]['SupportedSystems']['Brand']


def Brand_Model_Object(idx):
    """
    return dict value at certain level
    """
    return dictt[idx]['SupportedSystems']['Brand']['Model']


def Brand_Model_Display_Text_Object(idx):
    """
    return dict value at certain level
    """
    return dictt[idx]['SupportedSystems']['Brand']['Model']['Display']['#text']


def Brand_Model_Object_D(idx, _DD):
    """
    return dict value at certain level
    """
    return Brand_Model_Object(idx)[_DD]['Display']['#text']


def Brand_Object_D_M(idx, _DD):
    """
    return dict value at certain level
    """
    return Brand_Object(idx)[_DD]['Model']


def Brand_Object_D_M_complete(idx, _DD):
    """
    return dict value at certain level
    """
    return Brand_Object(idx)[_DD]['Model']['Display']['#text']


def Brand_Object_D_M_L_complete(idx, _DD, _EE):
    """
    return dict value at certain level
    """
    return Brand_Object_D_M(idx, _DD)[_EE]['Display']['#text']


def Brand_Object_D(idx):
    """
    return dict value at certain level
    """
    return Brand_Object(idx)['Display']['#text']


def Brand_Object_L(_FF, idx):
    """
    return dict value at certain level
    """
    return Brand_Object(idx)[_FF]['Display']['#text']


# keep looping until dict is reached
# then read the value using the relevant key of the dict
def read_BiosPc_xml_Model():
    """
    """
    Model = []
    max_colNumList = []
    max_col_num = None
    for idx in range(len_dict):
        if is_dict(Brand_Object(idx)):
            if (is_dict(Brand_Model_Object(idx))):
                # default
                get_model = [Brand_Model_Display_Text_Object(idx)]
                Model.append(get_model)
                # print(idx+1, get_model)

            elif (is_list(Brand_Model_Object(idx))):
                BMOL = []
                for _dd in range(len(Brand_Model_Object(idx))):
                    BMOL.append(Brand_Model_Object_D(idx, _dd))
                max_colNumList.append(_dd+1)  # end value of will be max
                # print(idx+1, BMOL)
                get_model = BMOL
                Model.append(get_model)
            else:
                print(idx+1, "This is Brand-Model level exception")

        elif (is_list(Brand_Object(idx))):
            BOL = []
            for _dd in range(len(Brand_Object(idx))):
                if (is_dict(Brand_Object_D_M(idx, _dd))):
                    get_model = Brand_Object_D_M_complete(
                        idx, _dd)  # "unlist here"
                    BOL.append(get_model)
                elif (is_list(Brand_Object_D_M(idx, _dd))):
                    for _ee in range(len(Brand_Object_D_M(idx, _dd))):
                        get_model = Brand_Object_D_M_L_complete(
                            idx, _dd, _ee)  # "unlist here"
                        BOL.append(get_model)
                    max_colNumList.append(_ee+1)  # end value of will be max
            get_model = BOL
            Model.append(get_model)

        else:
            print(idx+1, "This is Brand level exception")
    max_colNumList.sort()
    max_col_num = max_colNumList[-1]
    return Model, max_col_num


def read_BiosPc_xml_Brand():
    """
    """
    Brand = []
    max_colNumList = []
    max_col_num = None
    for idx in range(len_dict):
        if is_dict(Brand_Object(idx)):
            # print(idx+1, Brand_Object_D(idx))
            get_brand = [Brand_Object_D(idx)]
            Brand.append(get_brand)

        elif is_list(Brand_Object(idx)):
            BOBL = []
            for _ff in range(len(Brand_Object(idx))):
                get_brand = Brand_Object_L(_ff, idx)
                BOBL.append(get_brand)
            # print(idx+1, BOBL)
            get_brand = BOBL
            Brand.append(get_brand)
            max_colNumList.append(_ff+1)
        else:
            print(idx+1, "This is an exception")
    max_colNumList.sort()
    max_col_num = max_colNumList[-1]
    return Brand, max_col_num


def read_BiosPc_xml_schemaVersion():
    schemaVersion = []
    for idx in range(len_dict):
        get_val = dictt[idx]['@schemaVersion']
        # print(idx+1, get_val)
        schemaVersion.append(get_val)
    return schemaVersion


def read_BiosPc_xml_releaseID():
    releaseID = []
    for idx in range(len_dict):
        get_val = dictt[idx]['@releaseID']
        # print(idx+1, get_val)
        releaseID.append(get_val)
    return releaseID


def read_BiosPc_xml_releaseDate():
    releaseDate = []
    for idx in range(len_dict):
        get_val = dictt[idx]['@releaseDate']
        # print(idx+1, get_val)
        releaseDate.append(get_val)
    return releaseDate


def read_BiosPc_xml_vendorVersion():
    vendorVersion = []
    for idx in range(len_dict):
        get_val = dictt[idx]['@vendorVersion']
        # print(idx+1, get_val)
        vendorVersion.append(get_val)
    return vendorVersion


def read_BiosPc_xml_myVersion():
    myVersion = []
    for idx in range(len_dict):
        get_val = dictt[idx]['@myVersion']
        # print(idx+1, get_val)
        myVersion.append(get_val)
    return myVersion


def read_BiosPc_xml_packageType():
    packageType = []
    for idx in range(len_dict):
        get_val = dictt[idx]['@packageType']
        # print(idx+1, get_val)
        packageType.append(get_val)
    return packageType


def read_BiosPc_xml_path():
    path = []
    for idx in range(len_dict):
        get_val = dictt[idx]['@path']
        # print(idx+1, get_val)
        path.append(get_val)
    return path


def read_BiosPc_xml_packageID():
    packageID = []
    for idx in range(len_dict):
        get_val = dictt[idx]['@packageID']
        # print(idx+1, get_val)
        packageID.append(get_val)
    return packageID


def read_BiosPc_xml_dateTime():
    dateTime = []
    for idx in range(len_dict):
        get_val = dictt[idx]['@dateTime']
        # print(idx+1, get_val)
        dateTime.append(get_val)
    return dateTime


def read_BiosPc_xml_size():
    size = []
    for idx in range(len_dict):
        get_val = dictt[idx]['@size']
        # print(idx+1, get_val)
        size.append(get_val)
    return size


def read_BiosPc_xml_identifier():
    identifier = []
    for idx in range(len_dict):
        get_val = dictt[idx]['@identifier']
        # print(idx+1, get_val)
        identifier.append(get_val)
    return identifier


def read_BiosPc_xml_Name():
    Name = []
    for idx in range(len_dict):
        get_val = dictt[idx]['Name']['Display']['#text']
        # print(idx+1, get_val)
        Name.append(get_val)
    return Name


def read_BiosPc_xml_ComponentType():
    ComponentType = []
    for idx in range(len_dict):
        get_val = dictt[idx]['ComponentType']['@value']
        # print(idx+1, get_val)
        ComponentType.append(get_val)
    return ComponentType


def read_BiosPc_xml_Description():
    Description = []
    for idx in range(len_dict):
        get_val = dictt[idx]['Description']['Display']['#text']
        # print(idx+1, get_val)
        Description.append(get_val)
    return Description


def read_BiosPc_xml_Category():
    Category = []
    for idx in range(len_dict):
        get_val = dictt[idx]['Category']['Display']['#text']
        # print(idx+1, get_val)
        Category.append(get_val)
    return Category


def read_BiosPc_xml_SupportedDevices():
    SupportedDevices = []
    for idx in range(len_dict):
        get_val = dictt[idx]['SupportedDevices']['Device']['Display']
        # print(idx+1, get_val)
        SupportedDevices.append(get_val)
    return SupportedDevices


def read_BiosPc_xml_ImportantInfo():
    ImportantInfo = []
    for idx in range(len_dict):
        get_val = dictt[idx]['ImportantInfo']['@URL']
        # print(idx+1, get_val)
        ImportantInfo.append(get_val)
    return ImportantInfo


def read_BiosPc_xml_Criticality():
    Criticality = []
    for idx in range(len_dict):
        get_val = dictt[idx]['Criticality']['Display']['#text']
        # print(idx+1, get_val)
        Criticality.append(get_val)
    return Criticality


def read_BiosPc_xml_AllTags(dictt, len_dict):
    """[summary]

    Returns:
        [bundle of lists]: [bundle of lists of tags,
        tag values appended in a list]
    """
    try:
        model, _ = read_BiosPc_xml_Model()
        brand, _ = read_BiosPc_xml_Brand()
        schemaVersion = read_BiosPc_xml_schemaVersion()
        releaseID = read_BiosPc_xml_releaseID()
        releaseDate = read_BiosPc_xml_releaseDate()
        vendorVersion = read_BiosPc_xml_vendorVersion()
        myVersion = read_BiosPc_xml_myVersion()
        packageType = read_BiosPc_xml_packageType()
        path = read_BiosPc_xml_path()
        packageID = read_BiosPc_xml_packageID()
        dateTime = read_BiosPc_xml_dateTime()
        size = read_BiosPc_xml_size()
        identifier = read_BiosPc_xml_identifier()
        Name = read_BiosPc_xml_Name()
        ComponentType = read_BiosPc_xml_ComponentType()
        Description = read_BiosPc_xml_Description()
        Category = read_BiosPc_xml_Category()
        SupportedDevices = read_BiosPc_xml_SupportedDevices()
        ImportantInfo = read_BiosPc_xml_ImportantInfo()
        Criticality = read_BiosPc_xml_Criticality()
        print("\n------All tags read-------")
        return model, brand, schemaVersion, releaseID, releaseDate,\
            vendorVersion, myVersion, packageType, path, packageID,\
            dateTime, size, identifier,\
            Name, ComponentType, Description, Category, SupportedDevices,\
            ImportantInfo, Criticality

    except Exception:
        print("One or more tags could not be read")


def list_to_DF(listInEachTag, String):
    """[summary]

    Args:
        listInEachTag ([type]): [list of values parsed from a tag]
        String ([type]): [tag as string]

    Returns:
        [type]: [dataframe of list]
    """
    try:
        df = pd.DataFrame(listInEachTag)
    except Exception:
        df = pd.DataFrame()
    nc = df.shape[1]
    assert nc > 0, "This is not a valid dataframe"
    cols = [String + str(i + 1) for i in range(nc)]
    df.columns = cols
    # remove 'None' from dataframe before concat
    mask = df.applymap(lambda x: x is None)
    None_cols = df.columns[(mask).any()]
    for cc in df[None_cols]:
        df.loc[mask[cc], cc] = ""
    if nc > 1:
        # concat columns
        df[String] = df[cols].apply(
            lambda row: ";".join(row.values.astype(str)), axis=1)
        # remove multiple ";" from the string
        df[String] = df[String].str.replace('(;){2,6}', '')
        # remove ";" from the end of the string
        df[String] = df[String].str.replace('(;)$', '')
        # remove trailing spaces
        df[String] = df[String].str.strip()
    for col in df.columns:
        # upcase
        df[col] = df[col].str.upper()
        # remove forward slash and one space
        df[col] = df[col].str.replace("\\/ ", ";")
        # remove "System BIOS"
        df[col] = df[col].apply(
            lambda c: c.replace("SYSTEM BIOS", ""))
        # remove new line character
        df[col] = df[col].apply(
            lambda c: " ".join(e.strip('\n') for e in c.split()))
        # remove trailing spaces
        df[col] = df[col].str.strip()
        if ("name" in str(col).lower()):
            # remove "my " from name columns
            df[col] = df[col].apply(lambda c: c.replace("my ", ""))\
                             .apply(
                                 lambda c: c.replace(
                                     ",", "/") if " AND " in c else c)\
                             .apply(lambda c: c.replace(" AND ", "/"))\
                             .apply(lambda c: c.replace("BIOS", ""))\
                                 .apply(lambda c: c.replace("2-IN-1", "2N1"))\
                             .str.strip()
    return df


def xml_to_DF(subdir, filename):
    """[summary]

    Args:
        subdir (String): [Path to input xml file]
        filename ([String]): [input xml file]

    Returns:
        [type]: [convert xml file to dataframe]
    """
    xmlDocument = os.path.join(subdir, filename)
    with open(xmlDocument, 'rb') as f:
        try:
            xmlContent = xmltodict.parse(f)
            _d = json.loads(json.dumps(xmlContent))
        except xmltodict.expat.ExpatError:
            pass

    LV1 = 'Manifest'
    LV2 = 'SoftwareComponent'

    global dictt
    global len_dict

    dictt = _d[LV1][LV2]
    len_dict = (len(dictt))

    model, brand, schemaVersion, releaseID, releaseDate, vendorVersion,\
        myVersion, packageType, path, packageID, dateTime, size, identifier,\
        Name, ComponentType, Description, Category, SupportedDevices,\
        ImportantInfo, Criticality = read_BiosPc_xml_AllTags(dictt, len_dict)

    list_of_tags = ['model', 'brand', 'schemaVersion',
                    'releaseID', 'releaseDate', 'vendorVersion',
                    'myVersion', 'packageType', 'path',
                    'packageID', 'dateTime', 'size', 'identifier',
                    'Name', 'ComponentType', 'Description',
                    'Category', 'SupportedDevices',
                    'ImportantInfo', 'Criticality']

    cat_DF = pd.DataFrame()
    for _t in range(len(list_of_tags)):
        eachTag = list_of_tags[_t]
        tmp = list_to_DF(eval(eachTag), eachTag)
        cat_DF = pd.concat([cat_DF, tmp], axis=1)
    mask = cat_DF.applymap(lambda x: x is None)
    cols = cat_DF.columns[(mask).any()]
    for col in cat_DF[cols]:
        cat_DF.loc[mask[col], col] = ''
    return cat_DF


def expand_df(DF, column, delim):
    """[explode column values into
    separate row input dataframe]

    Args:
        DF ([Dataframe]): [input dataframe]
        column ([column]): [dataframe column]
        delim ([char]): [delimiter]

    Returns:
        [dataframe]: [Splits delimited column values into
        separate rows under the same column name]
    """
    return (DF.set_index(DF.columns.drop(column, 1).tolist())
            [column].str.split(delim, expand=True)
            .stack()
            .reset_index()
            .rename(columns={0: column})
            .loc[:, DF.columns])


def compareWStandard(std_series, to_ds, to_col):
    """[summary]

    Args:
        std_series ([series]): [series of standard names]
        to_ds ([dataframe]): [dataframe to standardize]
        to_col ([string]): [dataframe column to standardize]

    Returns:
        [dataframe]: [dataframe having "col + _match"
        as return column]
    """
    comp_list = np.unique(np.array(std_series))
    comp_list = list(comp_list)
    for it, row in to_ds.iterrows():
        list_item = row[to_col]
        output, _ = process.extractOne(
            list_item,
            comp_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=0.8)

        to_ds.loc[it, to_col + '_match'] = output

    return to_ds


def process_BiosPc(dat, col):
    """[summary]

    Args:
        dat ([dataframe]): [input dataframe which needs cleaning]
        col ([string]): [column to extract standard names from]

    Returns:
        [dataframe]: [output dataframe will have
        new column by the name: "input colname + _Rev"
        having standardized names]
    """
    dat_exp = expand_df(dat, column=col, delim=';')

    tmp = pd.DataFrame(dat_exp[col].str.split('/', expand=True))
    tmp.rename(
        columns={0: "first"}, inplace=True)

    first_String = tmp['first'].apply(lambda row: row.split(' ')[0])
    first_String.name = "first_String"

    tmp = pd.concat([tmp, first_String], axis=1)

    r_cols = [c for c in tmp.columns if c not in ['first', 'first_String']]

    for _tt, row in tmp.iterrows():
        for r_cols_i in r_cols:
            if (row[r_cols_i] is not None):
                if not (is_LOB(row[r_cols_i].split()[0])):
                    tmp.loc[_tt, r_cols_i] = row['first_String'] + \
                                            " " + row[r_cols_i]
                else:
                    tmp.loc[_tt, r_cols_i] = row[r_cols_i]
            else:
                continue

    col_rev = col + "_Rev"
    tmp.loc[:, col_rev] = ""
    for _uu, row in tmp.iterrows():
        tmp.loc[_uu, col_rev] = row['first']
        for col in (r_cols):
            if row[col] is not None:
                tmp.loc[_uu, col_rev] = \
                    row[col_rev] + ";" + row[col]

    dat_exp_rev = pd.concat([dat_exp, tmp[[col_rev]]], axis=1)
    dat_rev_exp_d1 = expand_df(dat_exp_rev, column=col_rev, delim=';')
    dat_rev_exp_d2 = expand_df(
        dat_rev_exp_d1, column=col_rev, delim=', ')

    return dat_rev_exp_d2


def main():

    subdir = "../Data/"
    filename = "BiosPc.xml"

    # path to export the files
    csvFile = '../Data/out/BiosPc.csv'
    csvFile = None

    standard_df = get_standard_product()
     = get__LOGS()

    DF = xml_to_DF(subdir, filename)

    name_rev_exp_d2 = process_BiosPc(DF, "Name1")

    print("\n----Find best matches for BiosPC data----")
    Name1_Rev_match = (
            compareWStandard(
                standard_df["Product_line"], name_rev_exp_d2, "Name1_Rev")
    )

    print("\n----Merge BiosPc and Logs data----")
    _to_BiosPc = pd.merge(, Name1_Rev_match,
                               left_on="_BRAND",
                               right_on="Name1_Rev_match", how="left")

    print("Number of Unique assets in Logs data for given duration: {}"
          .format(['_ID'].nunique()))

    print("Number of Unique assets in merged data: {}"
          .format(_to_BiosPc['_ID'].nunique()))

    print("Number of Unique Brands in Logs data for given duration: {}"
          .format(['_BRAND'].nunique()))

    print("Number of Unique Brands in merged data: {}"
          .format(_to_BiosPc['_BRAND'].nunique()))

    if (csvFile):
        _to_BiosPc.to_csv(csvFile, index=False, header=True)


if __name__ == '__main__':

    import numpy as np
    import pandas as pd
    import xmltodict
    import os
    import json
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    from IPython.core.display import display

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    # These variable must exist in the global namespace first
    dictt = None
    len_dict = None

    try:
        main()
    except Exception:
        import traceback
        print(traceback.format_exc())
        print("Exception occured")
