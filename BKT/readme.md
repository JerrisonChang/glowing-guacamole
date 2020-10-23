The module for BKT that I use can be found here:
https://iedms.github.io/standard-bkt/
the predicthmm and trainhmm are the compiled executables from the repository.
If you are using Linux or Windows, you'll need to compile a new executables from the source code in the repository.

training model:
```
./BKT/trainhmm -s 1.2 -d ~ -m 1 -p 1 ./BKT/inputs/clean_data\ KC\ \(Original\)_training.txt ./BKT/model/model.txt ./BKT/model/predict.txt
```

predicting:
```
./BKT/predicthmm -p 1 ./BKT/inputs/clean_data\ KC\ \(Original\)_predicting.txt ./BKT/model/model.txt ./BKT/model/predict_from_predict.txt
```

