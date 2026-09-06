## Como executar
python -m representation_extraction.extractors.tfidf
python -m representation_extraction.extractors.bow

## Treinamento e teste
Base de treinamento: 70/30 de maneira aleatória. Guardar o teste pro fim.
Usar acurácia para comparar representações
Separar uma parte do comments_train.txt pra validação. O comments_test.txt rodar
apenas para TESTE.


## Pastas e arquivos
Pasta dataset/ contém os dados de treino e teste
Pasta results/ contém os resultados dos experimentos, em .csv sendo no formato: {metodo}.csv. 
Pasta results/test --> contém os resultados dos testes
Pasta results/validation --> contém os resultados da validação

compare.py Compara a acurácia entre os métodos de extração

## Métricas consideradas
* Acurácia de validação
* Acurácia de teste
* Tempo de execução