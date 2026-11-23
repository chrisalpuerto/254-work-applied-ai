| Vectorizer      | Stop-word removal   |   # features |   Test accuracy |   Macro precision |   Train time (s) |
|:----------------|:--------------------|-------------:|----------------:|------------------:|-----------------:|
| CountVectorizer | No                  |        64889 |          0.9095 |            0.9097 |             62.8 |
| TfidfVectorizer | No                  |        64889 |          0.9113 |            0.9113 |             64.7 |
| CountVectorizer | Yes                 |        64585 |          0.9088 |            0.909  |             61.2 |
| TfidfVectorizer | Yes                 |        64585 |          0.9105 |            0.9108 |             61.7 |

Best: TfidfVectorizer, stop-word removal = No
