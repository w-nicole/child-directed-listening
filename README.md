# child-grammar

Entire codebase was initialized from https://github.com/smeylan/child-directed-listening (approx. 11/19/22).


# Setup

These assume you are working on macOS, and have Anaconda installed. 

```conda create -n child-grammar```

Download and install the appropriate R package here: https://repo.miserver.it.umich.edu/cran/

```pip3 install -r requirements.txt```

Run ```R``` to get an R session.

Then, following the recommendation/code of [1], run the following:

```

install.packages("rlang")

install.packages("lazyeval")

install.packages("ggplot2")

```

Now, back in Terminal, run

```pip3 install matplotlib levenshtein scipy scikit-learn transformers pickle5 gdown```

Finally, following an adaptation of the original installation directions here [2]:

```pip3 install --user ipykernel; python3 -m ipykernel install --user --name=child-grammar```

# Running codebase

Run each of the notebooks in `tier_1_data_generation` in order.

Then, run `chmod u+x tier_2_scoring/data_unigram.sh; ./tier_2_scoring/data_unigram.sh` 

Upload the file `viewable_levdist_generated_glosses.csv` into Google Sheets and add a column `is_grammatical`. Delete the unlabeled index column, and give every row a score according to the description in the paper of grammaticality.

Then upload that file as `scored_levdist_generated_glosses.csv` into the same folder as the `viewable` csv file and run `chmod u+x tier_3_analyses/analyses.sh; ./tier_3_analyses/analyses.sh`.

The outputs will be in `outputs/experiments/full_scale/eval/n={number depending on your configuration}`.

[1] (11/19/22) https://community.rstudio.com/t/install-rlang-package-issue/84072/2

[2] (11/19/22) https://github.com/smeylan/child-directed-listening