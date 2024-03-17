#Project_1

#For train.py:

input: python train.py train.txt
output: создан файл model.torch и char_to_ix.json

#For test.py:

input: python test.py --char-to-ix char_to_ix.json model.torch test.txt
output: Предсказанное: е,
Test Loss: 6.6436
Предсказанное: а, 
Test Loss: 6.1221
Предсказанное: ч,
Test Loss: 6.3446
Предсказанное: ф,
Test Loss: 6.3663
Предсказанное: к,
Test Loss: 6.4642
Average Test Loss: 6.3882 (не понимаю почему тестовый loss такой большой, на обучении loss = 1.5)
