# Books — Active Learning: Priority Rating Queue

Your Hardcover **Want-to-Read** shelf (20 unrated books), ranked by **novelty** — kNN cosine distance from each book to your rated library in the book feature space. Rating the most novel books first teaches the model the most per rating. `Predicted` is the deployed **text-similarity** model (embedding-kNN over plot/author/description — skill 0.43 vs 0.24 for the old crowd-feature SVR) snapped to the 0.5★ grid, with its 80% conformal interval (±0.65, from the trained model's OOF residuals).

|   Rank | Title                                                                                         | Authors                                                    | Predicted (±0.5★)   |   Novelty |
|-------:|:----------------------------------------------------------------------------------------------|:-----------------------------------------------------------|:--------------------|----------:|
|      1 | The Locked Tomb Series, Harrow the Ninth, Gideon the Ninth and Nona the Ninth, Set of 3 Books | Tamsyn Muir                                                | 3.0 ±0.65           |    0.5328 |
|      2 | The Ballad of Black Tom                                                                       | Victor LaValle                                             | 3.5 ±0.65           |    0.4834 |
|      3 | The Cipher                                                                                    | Kathe Koja                                                 | 3.5 ±0.65           |    0.4729 |
|      4 | Diaspora                                                                                      | Greg Egan                                                  | 3.5 ±0.65           |    0.4713 |
|      5 | Slaughterhouse-Five                                                                           | Kurt Vonnegut                                              | 4.0 ±0.65           |    0.461  |
|      6 | North American Lake Monsters                                                                  | Nathan Ballingrud                                          | 4.0 ±0.65           |    0.4525 |
|      7 | Fahrenheit 451                                                                                | Ray Bradbury                                               | 3.5 ±0.65           |    0.418  |
|      8 | Don Quixote                                                                                   | Miguel de Cervantes Saavedra, Edith Grossman, Harold Bloom | 4.0 ±0.65           |    0.4063 |
|      9 | The Shadow over Innsmouth                                                                     | H.P. Lovecraft                                             | 4.5 ±0.65           |    0.4024 |
|     10 | The Martian                                                                                   | Andy Weir                                                  | 3.5 ±0.65           |    0.4018 |
|     11 | Portnoy's Complaint                                                                           | Philip Roth                                                | 3.5 ±0.65           |    0.399  |
|     12 | The Hundred Light-Year Diary                                                                  | Greg Egan                                                  | 3.0 ±0.65           |    0.3895 |
|     13 | 1984                                                                                          | George Orwell, Peter Hobley Davison, Shepard Fairey        | 3.5 ±0.65           |    0.3884 |
|     14 | The Picture of Dorian Gray                                                                    | Oscar Wilde, Robert Mighall                                | 4.0 ±0.65           |    0.3857 |
|     15 | The Immaculate Void                                                                           | Brian Hodge                                                | 3.5 ±0.65           |    0.3853 |
|     16 | The Hellbound Heart                                                                           | Clive Barker                                               | 3.5 ±0.65           |    0.383  |
|     17 | A Prayer for the Crown-Shy                                                                    | Becky Chambers                                             | 3.5 ±0.65           |    0.3652 |
|     18 | Train Dreams                                                                                  | Denis Johnson                                              | 4.0 ±0.65           |    0.3642 |
|     19 | I'm Thinking of Ending Things                                                                 | Iain Reid                                                  | 4.0 ±0.65           |    0.335  |
|     20 | The Fall                                                                                      | Albert Camus, Justin O'Brien                               | 4.0 ±0.65           |    0.31   |
