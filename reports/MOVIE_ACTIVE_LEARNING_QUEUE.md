# Movie Active-Learning Queue

**What to watch next to teach the model the most.** Ranked by `z(uncertainty) + z(novelty)`:

- **Uncertainty** = disagreement among the ensemble's base models. High = the model is torn and your rating settles it.
- **Novelty** = cosine distance (title+director+plot embedding) to your 5 nearest rated films. High = covers taste-space the model has few examples of.

_418 watchlist films scored. Top 30 shown._

| Film                                           |   Year |   Est. Rating |   Uncertainty |   Novelty |   Priority | Why                                       | URI                  |
|:-----------------------------------------------|-------:|--------------:|--------------:|----------:|-----------:|:------------------------------------------|:---------------------|
| Brokeback Mountain                             |   2005 |           4   |         2.101 |     0.484 |      4.289 | models disagree + unlike your rated films | https://boxd.it/2b2q |
| GOAT                                           |   2026 |           3.5 |         2.209 |     0.447 |      3.999 | models disagree + unlike your rated films | https://boxd.it/NvB0 |
| Mysterious Skin                                |   2004 |           3.5 |         1.805 |     0.487 |      3.556 | models disagree + unlike your rated films | https://boxd.it/1UsG |
| Pride & Prejudice                              |   2005 |           4   |         1.429 |     0.535 |      3.31  | models disagree + unlike your rated films | https://boxd.it/24u8 |
| La Chimera                                     |   2023 |           4   |         1.711 |     0.485 |      3.27  | models disagree + unlike your rated films | https://boxd.it/vzVi |
| 12 Years a Slave                               |   2013 |           4   |         1.529 |     0.512 |      3.219 | models disagree + unlike your rated films | https://boxd.it/2D2e |
| Neon Genesis Evangelion: The End of Evangelion |   1997 |           4   |         1.651 |     0.481 |      3.053 | models disagree + unlike your rated films | https://boxd.it/1GRM |
| Colorful                                       |   2010 |           3.5 |         2.051 |     0.408 |      2.977 | models disagree                           | https://boxd.it/4UG  |
| Heroic Times                                   |   1983 |           3.5 |         1.612 |     0.48  |      2.933 | models disagree + unlike your rated films | https://boxd.it/7aSK |
| Indiana Jones and the Temple of Doom           |   1984 |           3.5 |         1.651 |     0.47  |      2.885 | models disagree + unlike your rated films | https://boxd.it/2bb8 |
| Nimona                                         |   2023 |           4   |         1.783 |     0.447 |      2.873 | models disagree + unlike your rated films | https://boxd.it/Aiyc |
| The Long Goodbye                               |   1973 |           3.5 |         1.807 |     0.428 |      2.639 | models disagree + unlike your rated films | https://boxd.it/281O |
| The Devil's Backbone                           |   2001 |           3.5 |         1.968 |     0.4   |      2.635 | models disagree                           | https://boxd.it/28Li |
| Ne Zha                                         |   2019 |           3.5 |         2.007 |     0.393 |      2.619 | models disagree                           | https://boxd.it/mKSm |
| Wolfs                                          |   2024 |           3   |         1.321 |     0.499 |      2.474 | models disagree + unlike your rated films | https://boxd.it/x6zG |
| I Like Movies                                  |   2022 |           3.5 |         1.444 |     0.477 |      2.456 | models disagree + unlike your rated films | https://boxd.it/B3u2 |
| Gladiator                                      |   2000 |           4   |         1.647 |     0.442 |      2.431 | models disagree + unlike your rated films | https://boxd.it/2b9m |
| Straight Outta Compton                         |   2015 |           4   |         1.456 |     0.472 |      2.405 | models disagree + unlike your rated films | https://boxd.it/8rMc |
| Boyz n the Hood                                |   1991 |           3.5 |         1.433 |     0.472 |      2.343 | models disagree + unlike your rated films | https://boxd.it/29Ug |
| The Fabulous Baron Munchausen                  |   1962 |           3.5 |         1.126 |     0.522 |      2.321 | unlike your rated films                   | https://boxd.it/ZV2  |
| The Worst Person in the World                  |   2021 |           4   |         1.734 |     0.419 |      2.314 | models disagree + unlike your rated films | https://boxd.it/oyws |
| North by Northwest                             |   1959 |           4   |         1.727 |     0.42  |      2.308 | models disagree + unlike your rated films | https://boxd.it/2aRM |
| Seven Samurai                                  |   1954 |           4   |         1.734 |     0.419 |      2.305 | models disagree + unlike your rated films | https://boxd.it/2axi |
| The Wrestler                                   |   2008 |           4   |         1.413 |     0.472 |      2.293 | models disagree + unlike your rated films | https://boxd.it/1So8 |
| Cha Cha Real Smooth                            |   2022 |           3.5 |         1.608 |     0.439 |      2.286 | models disagree + unlike your rated films | https://boxd.it/uF8E |
| Extremely Unique Dynamic                       |   2024 |           3.5 |         1.357 |     0.48  |      2.261 | models disagree + unlike your rated films | https://boxd.it/Jkyc |
| Sound of Metal                                 |   2019 |           4   |         1.612 |     0.435 |      2.246 | models disagree + unlike your rated films | https://boxd.it/i7Q4 |
| Misery                                         |   1990 |           4   |         1.771 |     0.408 |      2.231 | models disagree                           | https://boxd.it/28gY |
| The Descent                                    |   2005 |           3.5 |         1.481 |     0.454 |      2.197 | models disagree + unlike your rated films | https://boxd.it/1YNy |
| Sid and Nancy                                  |   1986 |           3.5 |         1.478 |     0.452 |      2.151 | models disagree + unlike your rated films | https://boxd.it/1N6e |

---
### Least informative (model already confident + familiar)

| Film                                             |   Year |   Est. Rating |   Priority |
|:-------------------------------------------------|-------:|--------------:|-----------:|
| Khatta Meetha                                    |   2010 |           2.5 |     -4.785 |
| Kabir Singh                                      |   2019 |           2.5 |     -3.822 |
| Cocktail                                         |   2012 |           2.5 |     -3.509 |
| Pirates of the Caribbean: Dead Men Tell No Tales |   2017 |           2.5 |     -3.433 |
| The Sacrament                                    |   2013 |           2   |     -3.22  |
| The Last Thing Mary Saw                          |   2021 |           1.5 |     -3.175 |
| Badmaash Company                                 |   2010 |           2.5 |     -2.958 |
| Godzilla vs. Kong                                |   2021 |           3   |     -2.943 |
| Baazigar                                         |   1993 |           3.5 |     -2.831 |
| Hum Saath Saath Hain                             |   1999 |           3   |     -2.789 |
