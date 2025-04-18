@echo off
:: Script para executar múltiplas configurações de argumentos em paralelo

:: Caminho para o interpretador Python
set PYTHON_PATH=D:\Anaconda\envs\Festou\python.exe

:: Caminho para o script main_classif.py
set SCRIPT_PATH=d:\Repos\neurocomputing\main_classif.py

:: Configurações de argumentos
set ARGS1=--epochs 10 --hidden_size 3 --sim_time 100 --learning_rate 0.1 --gamma 1.0 --rce_learning_rate 0.5 --rbf_gamma 0.5
set ARGS2=--epochs 10 --hidden_size 5 --sim_time 200 --learning_rate 0.1 --gamma 1.0 --rce_learning_rate 0.5 --rbf_gamma 0.5
set ARGS3=--epochs 20 --hidden_size 5 --sim_time 200 --learning_rate 0.01 --gamma 0.5 --rce_learning_rate 0.1 --rbf_gamma 0.2
set ARGS4=--epochs 20 --hidden_size 3 --sim_time 200 --learning_rate 0.01 --gamma 0.5 --rce_learning_rate 0.1 --rbf_gamma 0.2
set ARGS5=--epochs 15 --hidden_size 4 --sim_time 150 --learning_rate 0.05 --gamma 0.8 --rce_learning_rate 0.3 --rbf_gamma 0.4
set ARGS6=--epochs 25 --hidden_size 7 --sim_time 200 --learning_rate 0.1 --gamma 0.8 --rce_learning_rate 0.3 --rbf_gamma 0.4

:: Executa cada configuração em um novo terminal
start cmd /k %PYTHON_PATH% %SCRIPT_PATH% %ARGS1%
start cmd /k %PYTHON_PATH% %SCRIPT_PATH% %ARGS2%
start cmd /k %PYTHON_PATH% %SCRIPT_PATH% %ARGS3%
start cmd /k %PYTHON_PATH% %SCRIPT_PATH% %ARGS4%
start cmd /k %PYTHON_PATH% %SCRIPT_PATH% %ARGS5%
start cmd /k %PYTHON_PATH% %SCRIPT_PATH% %ARGS6%


:: Fim do script
echo Todas as execuções foram iniciadas em terminais separados.
pause