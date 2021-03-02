# install
## 1. `apt-get install pdf2svg` for Ubuntu or `brew install pdf2svg` for Mac
## 2. pip install svg.path

PDF_ROOT="../datasets/pdf"
SVG_ROOT="../datasets/svg"
OUTPUT_ROOT="../datasets/json"
REFINE_ROOT="../datasets/refine_json"
CPU=4

python digital_extractor.py --root $PDF_ROOT \
                            --output $SVG_ROOT \
                            --pdf2svg \
                            --cpu $CPU
sleep 10
python digital_extractor.py --root $SVG_ROOT \
                            --output $OUTPUT_ROOT \
                            --cpu $CPU
sleep 10
python digital_extractor.py --root $OUTPUT_ROOT \
                            --output $REFINE_ROOT \
                            --refine \
                            --cpu $CPU
