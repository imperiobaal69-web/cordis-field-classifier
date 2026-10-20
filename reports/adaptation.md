# Threshold re-tuning on a small labelled Horizon Europe sample

Evaluated on 16,105 Horizon projects never used for tuning. Row 0 uses thresholds tuned on H2020 validation.

| model                          |   labelled Horizon projects |   micro_f1 |   macro_f1 |   precision |   recall |
|:-------------------------------|----------------------------:|-----------:|-----------:|------------:|---------:|
| finetune:bge-small-en-v1.5     |                           0 |      0.709 |      0.609 |       0.659 |    0.766 |
| finetune:bge-small-en-v1.5     |                         100 |      0.713 |      0.612 |       0.685 |    0.744 |
| finetune:bge-small-en-v1.5     |                         250 |      0.711 |      0.61  |       0.703 |    0.719 |
| finetune:bge-small-en-v1.5     |                         500 |      0.704 |      0.605 |       0.685 |    0.724 |
| finetune:bge-small-en-v1.5     |                        1000 |      0.715 |      0.61  |       0.697 |    0.733 |
| finetune:bge-small-en-v1.5     |                        4042 |      0.715 |      0.612 |       0.705 |    0.725 |
| tfidf_logreg                   |                           0 |      0.638 |      0.534 |       0.58  |    0.708 |
| tfidf_logreg                   |                         100 |      0.641 |      0.521 |       0.58  |    0.716 |
| tfidf_logreg                   |                         250 |      0.636 |      0.519 |       0.579 |    0.706 |
| tfidf_logreg                   |                         500 |      0.643 |      0.522 |       0.627 |    0.661 |
| tfidf_logreg                   |                        1000 |      0.645 |      0.517 |       0.621 |    0.671 |
| tfidf_logreg                   |                        4042 |      0.645 |      0.533 |       0.619 |    0.675 |
| embed_logreg:bge-small-en-v1.5 |                           0 |      0.556 |      0.438 |       0.484 |    0.653 |
| embed_logreg:bge-small-en-v1.5 |                         100 |      0.544 |      0.353 |       0.575 |    0.516 |
| embed_logreg:bge-small-en-v1.5 |                         250 |      0.553 |      0.418 |       0.498 |    0.622 |
| embed_logreg:bge-small-en-v1.5 |                         500 |      0.562 |      0.412 |       0.547 |    0.577 |
| embed_logreg:bge-small-en-v1.5 |                        1000 |      0.568 |      0.405 |       0.554 |    0.583 |
| embed_logreg:bge-small-en-v1.5 |                        4042 |      0.566 |      0.444 |       0.531 |    0.607 |
