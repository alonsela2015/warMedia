import semopy
from preprocessing import excel_column_number
import pandas as pd


if __name__ == "__main__":
    FILE_PATH = 'scaled_data_12.12.respondents.xlsx'
    df = pd.read_excel(FILE_PATH)

    phone_df = df.iloc[:, excel_column_number('AN'):excel_column_number('AS') + 1]

    model_desc = "Unaware =~"
    questions_to_x = {}
    x_to_questions = {}
    for i, col in enumerate(phone_df.columns):
        questions_to_x[col] = f"x{i}"
        x_to_questions[f"x{i}"] = col
        if not i:
            model_desc += f"x{i}"
        else:
            model_desc += f" + x{i}"

    # Changing columns names
    phone_df = phone_df.rename(columns=questions_to_x)

    model = semopy.Model(model_desc)

    results = model.fit(phone_df)
    print(results)
    print(model.inspect())