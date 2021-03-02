# install
## pip install pdfminer3k==1.3.1
## pip install pdf2image==1.11.0

PDF_ROOT="../datasets/pdf"
OCR_OUTPUT="../datasets/ocr_result"
CPU=4

CSV_PATH="../datasets/ocr_result/save_data_profile_v3.csv"
JSON_PATH="./raw_label_hierarchy.json"
SAVE_PATH="../datasets/ocr_result/ruijin_ocr_processed_v1.csv"

python OCR_reader_mp.py --root $PDF_ROOT \
                        --output $OCR_OUTPUT \
                        --num_workers $CPU
sleep 10
python process_ocr_csv.py --csv $CSV_PATH \
                          --json $JSON_PATH \
                          --save $SAVE_PATH