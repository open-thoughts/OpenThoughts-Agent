import os
import pandas as pd

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
except ImportError:
    print("Google API not installed. Run `pip install google-api-python-client`")
    print("Remote sheet functionality will not work.")
    Credentials, Request, InstalledAppFlow, build = None, None, None, None

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]


def read_local_sheet(sheet_path: str):
    df = pd.read_csv(sheet_path)
    return df


def write_local_sheet(df: pd.DataFrame, sheet_path: str):
    df.to_csv(sheet_path, index=False)


def write_to_sheet(
    df: pd.DataFrame, sheet_id: str, sheet_range: str, credentials_path: str
):
    if credentials_path is None:
        write_local_sheet(df, sheet_id)
    else:
        if Credentials is None:
            print(
                "Google API not installed. Run `pip install google-api-python-client`"
            )
            print("Remote sheet functionality will not work.")
            return
        write_to_remote_sheet(df, sheet_id, sheet_range, credentials_path)


def read_sheet(sheet_id: str, sheet_range: str, credentials_path: str = None):
    if credentials_path is None:
        if not os.path.exists(sheet_id):
            raise FileNotFoundError(f"Local path for sheet {sheet_id} does not exist.")
        return read_local_sheet(sheet_id)
    else:
        return read_remote_sheet(sheet_id, sheet_range, credentials_path)


def update_cell(
    string: str,
    col_name: str,
    sheet_id: str,
    df: pd.DataFrame,
    row_id: int,
    credentials_path: str = None,
):
    df.loc[row_id, col_name] = string
    write_to_sheet(df, sheet_id, "Sheet1!A1:Z1000", credentials_path)


def write_to_remote_sheet(
    df: pd.DataFrame, sheet_id: str, sheet_range: str, credentials_path: str
):
    """
    Write a DataFrame to a Google Sheet
    """

    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    # body with header
    body = {"values": [df.columns.tolist()] + df.values.tolist()}

    result = (
        sheet.values()
        .update(
            spreadsheetId=sheet_id,
            range=sheet_range,
            valueInputOption="RAW",
            body=body,
        )
        .execute()
    )
    print(f"{result.get('updatedCells')} cells updated.")
    os.remove("token.json")


def read_remote_sheet(
    sheet_id: str,
    sheet_range: str,
    credentials_path: str,
    token_path: str = "token.json",
):
    """
    Read a Google Sheet into a DataFrame
    """

    if Credentials is None:
        print("Google API not installed. Run `pip install google-api-python-client`")
        print("Remote sheet functionality will not work.")
        return

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=sheet_id, range=sheet_range).execute()
    # fill in the missing values
    values = result.get("values", [])

    if len(values[1]) < len(values[0]):
        values[1] += [""] * (len(values[0]) - len(values[1]))

    return pd.DataFrame(values[1:], columns=values[0])


# for index, row in experiments_df.iterrows():
#     args = row.to_dict()

# # if exp_config.sheet_id:
# #     experiments_df = read_sheet(exp_config.sheet_id, "Sheet1!A1:Z1000", exp_config.credentials_path)

# experiments_df = pd.DataFrame(columns=config.keys(), index=[0])
# experiments_df.loc[0] = config
