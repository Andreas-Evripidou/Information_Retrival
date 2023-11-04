
echo "Running IR engine with $1"

echo "Running without stemming and stopping"
nohup python3 IR_engine.py -o test.txt -w $1 
python3 eval_ir.py cacm_gold_std.txt test.txt 

echo "Running with stopping"
nohup python3 IR_engine.py -o test.txt -w $1 -s
python3 eval_ir.py cacm_gold_std.txt test.txt 

echo "Running with stemming"
nohup python3 IR_engine.py -o test.txt -w $1 -p
python3 eval_ir.py cacm_gold_std.txt test.txt

echo "Running with stemming and stopping"
nohup python3 IR_engine.py -o test.txt -w $1 -s -p
python3 eval_ir.py cacm_gold_std.txt test.txt 
