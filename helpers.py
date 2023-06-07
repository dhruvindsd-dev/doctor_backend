from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from os import listdir
from os.path import isfile, join
import jellyfish as jd
from pdf2image import convert_from_path
import boto3
from trp import Document
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
import os
import openai

print("before setting env")
os.environ[""] = ""
persist_directory = "/Users/dhruvinddev/development/hackathon/doctor_backend/chroma"
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 1}),
    return_source_documents=True,
)
openai.api_key = ""


chat_history = []


print("after setting env")


def get_chatbot_response(query):
    # query = "where can i do surgeries of cardiomyopathy for cheap"
    val = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, val["answer"]))
    print(val.keys())

    try:
        base_response = {
            "query": val["question"],
            "result": val["answer"],
            "source_documents": dict(val["source_documents"][0]),
        }
    except:
        base_response = {
            "query": val["question"],
            "result": val["answer"],
            "source_documents": dict(val["source_documents"]),
        }

    return base_response


def handle_ocr(file_path):
    # Specify your file path here
    images = convert_from_path(file_path)
    file_name = file_path.split("/")[-1].split(".")[0]

    mypath = "cleaned"
    for filename in os.listdir(mypath):
        file_path = os.path.join(mypath, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

    for i in range(len(images)):
        # Save pages as images in the pdf
        images[i].save("cleaned/" + file_name + "_page" + str(i) + ".jpg", "JPEG")

    # Save pages as images in the pdf

    # Extract the file names from the directory
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # AWS credentials
    access_key = "AKIAYGIFDC36DQZXYZUG"
    secret_key = "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"

    # Create a session using the access key and secret value
    session = boto3.Session(
        aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    # Medical names
    medical_names = [
        "haemoglobin",
        "total w.b.c count",
        "neutrophils",
        "lymphocytes",
        "monocytes",
        "eosinophils",
        "basophils",
        "rbc count",
        "P.C.V. (HCT)",
        "M.C.V",
        "M.C.H.",
        "M.C.H.C.",
        "R.D.W.-CV",
        "R.D.W.-SD",
        "Platelet Count",
        "MPV",
        "E.S.R",
        "Blood Sugar - Fasting(Plasma Glucose)",
        "Urine Sugar",
        "Serum Creatinine",
        "eGFR",
        "ALT (SGPT)",
        "Serum Cholesterol",
        "H.D.L. Cholesterol",
        "Non-HDL Cholesterol",
        "Direct LDL cholesterol",
        "Cholesterol/HDL Ratio",
        "Serum Triglycerides",
        "V.L.D.L. Cholesterol",
        "Vitamin B12 Level",
        "Triiodohyronine (T3)",
        "Thyroxine (T4)",
        "TSH (Ultrasensitive)",
        "Specific Gravity",
    ]

    # Document
    mdf = pd.DataFrame()

    for i in sorted(onlyfiles):
        print(i)
        documentName = mypath + "/" + i

        # Amazon Textract client
        textract = session.client("textract", region_name="ap-south-1")

        # Call Amazon Textract
        with open(documentName, "rb") as document:
            response = textract.analyze_document(
                Document={
                    "Bytes": document.read(),
                },
                FeatureTypes=["TABLES"],
            )

        doc = Document(response)

        df = pd.DataFrame(columns=["page", "table", "cell", "name"])
        for page in doc.pages:
            # Print tables
            for table in page.tables:
                for r, row in enumerate(table.rows):
                    loop = False
                    for c, cell in enumerate(row.cells):
                        if c == 0 or loop == True:
                            for val in medical_names:
                                if (
                                    jd.jaro_distance(
                                        str(cell.text).lower().strip(), val
                                    )
                                    >= 0.80
                                    or loop == True
                                ):
                                    loop = True
                                    df.loc[len(df.index)] = [i, r, c, str(cell.text)]
                                    break

        try:
            df = (
                df.drop("page", axis=1)
                .pivot(index="table", columns="cell", values="name")
                .reset_index(drop=True)
            )
            df.columns = ["test_name", "result", "unit", "type", "reference_interval"]
            mdf = mdf.append(df)
            # df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        except Exception as e:
            print(e)
            pass
        # Test Name
    df = pd.DataFrame(columns=["page", "table", "cell", "name"])

    for page in doc.pages:
        # Print tables
        for table in page.tables:
            for r, row in enumerate(table.rows):
                loop = False
                for c, cell in enumerate(row.cells):
                    if c == 0 or loop == True:
                        for val in medical_names:
                            if (
                                jd.jaro_distance(str(cell.text).lower().strip(), val)
                                >= 0.80
                                or loop == True
                            ):
                                loop = True
                                df.loc[len(df.index)] = [i, r, c, str(cell.text)]
                                break

    # Return the extracted data as a JSON response
    response_data = {
        "message": "Extraction complete.",
        "medical_data": mdf.to_dict(),
        "test_names": df.to_dict(),
    }
    mdf = mdf.reset_index(drop=True)
    mdf_string = mdf.to_string(index=False)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "There is a medical document we would paste as an dataframe. Analyse it and think as an medical practioner. Provide your input, surgeries and suggessitions.",
            },
            {"role": "user", "content": f""" Dataframe "{mdf_string}" """},
        ],
    )

    result = ""
    for choice in response.choices:
        result += choice.message.content

    print(result)
    res = {"string": result, "mdf_json": mdf.to_dict(orient="records")}
    return res
