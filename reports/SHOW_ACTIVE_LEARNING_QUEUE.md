# TV Shows — Active Learning: What to Watch Next

Your Serializd **watchlist** (55 unwatched shows), ranked by **information gain** = z(uncertainty) + z(novelty). *Uncertainty* is the disagreement (std) across the TV ensemble's base learners (XGB, SVR, CatBoost, ordinal); *novelty* is the cosine distance from each show to your 5 nearest rated shows in the MiniLM→PCA text space. Rating the top shows first teaches the TV model the most per watch. `predicted_rating` is the stacking ensemble's guess (0.5★ grid) — informational, not the ranking key.

|   rank | name                             |   year |   predicted_rating |   uncertainty |   novelty |   priority_score | why                                       |
|-------:|:---------------------------------|-------:|-------------------:|--------------:|----------:|-----------------:|:------------------------------------------|
|      1 | Elite                            |   2018 |                2.5 |         0.58  |     0.389 |            4.46  | models disagree                           |
|      2 | Smallville                       |   2001 |                3   |         0.44  |     0.443 |            3.683 | models disagree + unlike your rated shows |
|      3 | SPY x FAMILY                     |   2022 |                4   |         0.191 |     0.571 |            2.71  | unlike your rated shows                   |
|      4 | Scavengers Reign                 |   2023 |                4   |         0.277 |     0.483 |            2.455 | models disagree + unlike your rated shows |
|      5 | My Adventures with Superman      |   2023 |                4   |         0.255 |     0.461 |            1.937 | models disagree + unlike your rated shows |
|      6 | Freaks and Geeks                 |   1999 |                4   |         0.293 |     0.408 |            1.631 | models disagree + unlike your rated shows |
|      7 | The Sandman                      |   2022 |                4   |         0.24  |     0.412 |            1.112 | models disagree + unlike your rated shows |
|      8 | The Chair Company                |   2025 |                3.5 |         0.291 |     0.371 |            1.107 | models disagree                           |
|      9 | Fargo                            |   2014 |                3.5 |         0.272 |     0.382 |            1.064 | models disagree                           |
|     10 | Legion                           |   2017 |                3   |         0.105 |     0.51  |            0.974 | unlike your rated shows                   |
|     11 | Heartstopper                     |   2022 |                4.5 |         0.236 |     0.398 |            0.885 | moderate                                  |
|     12 | Dexter: Original Sin             |   2024 |                4   |         0.218 |     0.411 |            0.859 | unlike your rated shows                   |
|     13 | The Great                        |   2020 |                3.5 |         0.187 |     0.435 |            0.855 | unlike your rated shows                   |
|     14 | Futurama                         |   1999 |                3.5 |         0.251 |     0.384 |            0.848 | models disagree                           |
|     15 | Dexter: New Blood                |   2021 |                3.5 |         0.275 |     0.362 |            0.821 | models disagree                           |
|     16 | Dark Matter                      |   2024 |                3.5 |         0.21  |     0.413 |            0.8   | unlike your rated shows                   |
|     17 | Widow's Bay                      |   2026 |                4   |         0.222 |     0.385 |            0.56  | moderate                                  |
|     18 | Superman & Lois                  |   2021 |                3   |         0.072 |     0.502 |            0.503 | unlike your rated shows                   |
|     19 | Lupin                            |   2021 |                3   |         0.311 |     0.308 |            0.485 | models disagree                           |
|     20 | Ash vs Evil Dead                 |   2015 |                4   |         0.137 |     0.446 |            0.453 | unlike your rated shows                   |
|     21 | Twin Peaks                       |   1990 |                4.5 |         0.254 |     0.343 |            0.338 | models disagree                           |
|     22 | The Rookie                       |   2018 |                3.5 |         0.077 |     0.476 |            0.215 | unlike your rated shows                   |
|     23 | Fullmetal Alchemist: Brotherhood |   2009 |                4.5 |         0.157 |     0.405 |            0.121 | moderate                                  |
|     24 | Luther                           |   2010 |                4   |         0.207 |     0.359 |            0.044 | moderate                                  |
|     25 | Frieren: Beyond Journey's End    |   2023 |                4.5 |         0.135 |     0.412 |           -0.022 | unlike your rated shows                   |
|     26 | Shrinking                        |   2023 |                4   |         0.193 |     0.361 |           -0.081 | moderate                                  |
|     27 | Utopia                           |   2013 |                4   |         0.151 |     0.388 |           -0.159 | moderate                                  |
|     28 | The Bear                         |   2022 |                4   |         0.175 |     0.361 |           -0.267 | moderate                                  |
|     29 | Reacher                          |   2022 |                3.5 |         0.151 |     0.378 |           -0.303 | moderate                                  |
|     30 | Superstore                       |   2015 |                3.5 |         0.282 |     0.269 |           -0.36  | models disagree                           |
|     31 | Succession                       |   2018 |                4.5 |         0.146 |     0.375 |           -0.396 | moderate                                  |
|     32 | Justice League                   |   2001 |                4   |         0.306 |     0.244 |           -0.425 | models disagree                           |
|     33 | Teen Titans                      |   2003 |                3.5 |         0.177 |     0.347 |           -0.431 | moderate                                  |
|     34 | Parasyte -the maxim-             |   2014 |                3.5 |         0.17  |     0.353 |           -0.431 | moderate                                  |
|     35 | Mad Men                          |   2007 |                4   |         0.101 |     0.408 |           -0.444 | unlike your rated shows                   |
|     36 | The Summer Hikaru Died           |   2025 |                4   |         0.107 |     0.403 |           -0.452 | moderate                                  |
|     37 | Everybody Hates Chris            |   2005 |                4   |         0.151 |     0.36  |           -0.543 | moderate                                  |
|     38 | Dexter: Resurrection             |   2025 |                4   |         0.062 |     0.43  |           -0.566 | unlike your rated shows                   |
|     39 | Welcome to the N-H-K             |   2006 |                4   |         0.091 |     0.406 |           -0.579 | moderate                                  |
|     40 | Hannibal                         |   2013 |                4   |         0.143 |     0.363 |           -0.588 | moderate                                  |
|     41 | Harley Quinn                     |   2019 |                3.5 |         0.16  |     0.348 |           -0.61  | moderate                                  |
|     42 | Common Side Effects              |   2025 |                4.5 |         0.171 |     0.33  |           -0.732 | moderate                                  |
|     43 | The IT Crowd                     |   2006 |                4   |         0.134 |     0.359 |           -0.746 | moderate                                  |
|     44 | 3 Body Problem                   |   2024 |                3   |         0.252 |     0.263 |           -0.759 | models disagree                           |
|     45 | Russian Doll                     |   2019 |                3.5 |         0.095 |     0.375 |           -0.949 | moderate                                  |
|     46 | The Apothecary Diaries           |   2023 |                4.5 |         0.177 |     0.283 |           -1.293 | moderate                                  |
|     47 | Kaguya-sama: Love Is War         |   2019 |                4   |         0.191 |     0.253 |           -1.552 | moderate                                  |
|     48 | Tokyo Ghoul                      |   2014 |                3.5 |         0.099 |     0.317 |           -1.677 | low info (confident + familiar)           |
|     49 | Doctor Who                       |   2005 |                4   |         0.12  |     0.297 |           -1.729 | low info (confident + familiar)           |
|     50 | FLCL                             |   2000 |                3.5 |         0.115 |     0.299 |           -1.752 | low info (confident + familiar)           |
|     51 | Yellowjackets                    |   2021 |                3.5 |         0.103 |     0.302 |           -1.844 | low info (confident + familiar)           |
|     52 | Akame ga Kill!                   |   2014 |                3.5 |         0.153 |     0.241 |           -2.122 | moderate                                  |
|     53 | Beck: Mongolian Chop Squad       |   2004 |                4   |         0.161 |     0.231 |           -2.171 | moderate                                  |
|     54 | Solo Leveling                    |   2024 |                4   |         0.179 |     0.212 |           -2.217 | moderate                                  |
|     55 | The Disastrous Life of Saiki K.  |   2016 |                4   |         0.068 |     0.264 |           -2.719 | low info (confident + familiar)           |
