from test_vectorization import main
from tools.svg_conversion import data_convert_to_absolute
from flask import Flask, request
import os

app = Flask(__name__)

@app.route('/enqueue-vectorization', methods=['POST'])
def upload():
    # receive img
    file = request.files['file']
    filepath = os.path.join("sample_inputs","clean_line_drawings",file.filename)
    # write to file
    with open(filepath, "wb") as f:
        f.write(file.getbuffer())
    # run vectorization model
    main(file.filename)
    # save as svg
    filenameNoType = os.path.basename(filepath)
    filenameNoType = os.path.splitext(filenameNoType)[0]
    data_convert_to_absolute(f"outputs/sampling/clean_line_drawings__pretrain_clean_line_drawings/seq_data/{filenameNoType}_0.npz","single")
    # read svg into string
    resultpath = os.path.join("outputs","sampling","clean_line_drawings__pretrain_clean_line_drawings","seq_data",f"{filenameNoType}_0","single.svg")
    lines = ""
    with open(resultpath) as f:
        lines = f.read()
    # return svg as string response
    return {'response': lines}
    
if __name__ == '__main__':
    app.run(debug=True)
