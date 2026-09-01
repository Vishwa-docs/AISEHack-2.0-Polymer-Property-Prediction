Anti AI Writing Skills

https://mail.google.com/mail/u/0/#inbox/FMfcgzQhWBlfZfDLHwbqgvGRqkmDjKTm
# Recreation Prompt

Deliver to the user's Mac final_submissions/ ONE completely standalone .py submission file that is written as REAL Python functions/classes only — NO TOOL_SOURCES dicts, NO VARIANT_CONFIGS/RECIPE JSON blobs, NO sha256/hash computation anywhere in the file, NO base64, NO tar/gzip, NO embedded archives, NO reference to previous runs, previous bundles, experiment records, or any precomputed CSV at runtime. The file takes ONLY the official competition dataset (train.csv/test.csv/PI1M.csv), regenerates every artifact (features, models, OOF, base compound chain, final arms) as actual Python functions called from a main() in dependency order, with every configuration value (weights, alphas, damp, spread scale, medians, seeds) written as literal Python constants in the file, and writes the 4940-row submission.csv that scores >= 0.9041. Someone with only the .py + official dataset must reproduce the score. No oracle/assisted wording anywhere. (2) The generated submission.csv. (3) Updated final_submissions/README.md with the verified score, architecture, and all hyperparameters — no oracle word, no hash listings. DO NOT run again until the user has verified and approved the file structure.

No intermediate csvs from previous runs can be called and used. If any is there like that, you need to trace them, find the code that generated them and then put that code in the py file. The blends also, same thing. We need to first generate the intermediate CSVs in the py file itself, and then use those for remaining, until we get the final 0.904 score.

Continue this trace and give me the py file for verification first, then you can begin testing adn verification of it after my approval.

# New Experiment Scaffolding
Read AGENTS.md and CONTEXT.md for context of the event and what we are trying to do here, and how to connect to GPU laptop.

Now in score_discrepancy/, there is another AGENTS.md. The goal for us is to maximize the R2 score against the final_oracle.csv. So far, if you read the files, we have achieved only a score of 0.891 on the private leaderboard (Or 0.900 Oracle Score). This is where you come in.

I need you to create a file called PLAN.md and put it inside the score_discrepancy/ folder. The goal is to run a few experiments, and reach an oracle score of 0.935 at least. 

To help you, there are a few things given : 

1. Phase 2 runs and score (These are not exactly accuract because the oracle score is higher than needed, but it can give you an idea). To access them, read AGENTS.md on the repo root to find our some experiments done. Some are also there in TRIALS.md file

2. 150 Experiments : In GPU Laptop, /Desktop/r3_runtime/Phase_2/ : I ran about a 150 experiments. There are some results here as well

3. score_discrepancy/NEW_EXPERIMENTS.md : I had asked other agents to give me a plan for a few experiments.

Apart from this, the organizers have given a new dataset for this round, which has 5.97 million unique molecular structures (Dataset/smile_r3.csv) which can be used for things like 

Self-supervised or unsupervised pretraining of molecular representations

SMILES-based representation learning

Feature enrichment and molecular embedding generation

Transfer learning to improve polymer property prediction models

Physics Informed Network

Different Polymer Structures : Can we find a relation



Analyse all of these methods (What is tried, what is not tried and new etc) and then proceed to the below.

What I need you to do is create a file called PLAN.md. This file will be given to another agent to do the coding, testing and everything else for the experiments. This is what I need : 

1. Goal : To get to a R2 score of 0.935+ on the Oracle/final_oracle.csv file

2. File Location : All files MUST be put in the GPU laptop, under Desktop/r3_runtime/Phase_3. NOTHING has to be there in my Mac laptop until the experiments are run and final results are in. Nothing should be there in scripts/ also. Once its there in GPU laptop and run and experiments are done, we can do analysis and move only required files.

3. Strict Instructions : We CANNOT use external datasets or anything like that. We HAVE to use what is provided in Round 3 only (The Pl1M, smile_r3.csv which is new, train.csv and test.csv). Oracle/final_oracle.csv is only used for final scoring and oracle assisted research (But not too obvious, we cannot let them know we are using oracle anywhere)

4. A list of experimetns, at least 200-300 of them. It can be techniques that we have not tried yet, something to improve the results etc. Make sure we are trying different things. Some of them can be long as well (Like using the 5.87 million ones will be difficult I am aware, so we can do some subset analysis and more, then run on the whole thing). ANything that gets to the goal we can try

5. Instructions to create a run.sh file (Similar to GPU Laptop's Desktop/r3_runtime/Phase_2/run.sh) so that everything can run in one caommand and all logs are stored. That way I can monitor and run without agents and then just analyse results.

Give me this PLAN.md file after doing through research and analysis, and update all isntructions that I have given as well. You can even do some eda befoere coming up with the final experiments plan


---
---

# Codebase Creation Prompt (Basic)
I need your help on a massive task. I need you to read AGENTS.md and CONTEXT.md first to get into on the task and context. It will also have instructions on how to connect to the GPU laptop. 

We are in Round 3 of the PPP contest now. I had run several experiments in Round 2, and the best pipeline was chosen and placed in my Mac's final_submissions/ folder. If you read my Mac's score_discrepancy/, you will see that this one scored 0.917 on the public leaderboard, but only 0.891 on the private LB, and so I updated the oracle and got the Oracle/final_oracle.csv (So round 2 experiment's score on the GPU laptop may not be that accurate).

For Round 3, I have worked on a few more experiments. I had run these in the below phases : 
+ Phase 2 (150 experiments) : GPU Laptop's ~/Desktop/r3_runtime/Phase_2/ will have the logs somewhere here
+ Phase 3 (250+ experiments) : GPU Laptop's ~/Desktop/r3_runtime/Phase_3/
+ Phase 5 : Experiments run again, on my Mac's /Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3/Phase5_Kiro_Score_Improvement/logs/phase5_summary.tsv and the same folder has the remaining experiments
+ Phase 5A : /Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3/Phase5A_Gap_Analysis/logs/phase5a_summary.tsv


These are the scores, although I dont know if they come close to or match the final_submissions/ score (Which scored 0.891 on the LB).

Here is the task for you : 
1. Create a folder called CODEBASE/ inside this repo.
2. We need to come up with the final pipeline

To do this, there is a strategy. We need not take the results at the face value. What I mean is that we have done a lot of analysis in Phase5A and we know a few things that work and don't. We have also the results of experiment by polymer type. So what we can do is make a compound pipeline.

Do do this, analyze the results of the phases and sort them. Find out which pipeline produces the best for each of the polymer properties, and then we can mix them. What I mean is that we can have an entire pipeline that is there JUST to fill in the values of Tg, then another pipeline for Ei etc for all 7. The final score is just the mean of individual R square values, so like this, we can maximize everything. So we can take whichever method works best for each of them, and then create individual pipelines (We know anyways the target in test.csv, so we can use train.csv fully and then each pipeline can only fill in the required files)

So first, sort it out and see, and tell me the mathematical best (The best from ALL experiments previously and phases) against the final_oracle.csv and what is the expectation. This is purely arithmetic, sort, see which can give the best possible results, and then how we can pool the pipelines together. Imputation is also good here, like if you can impute some entries from train.csv to test.csv (ONLY these, cant use oracle) for a few at least, and then predict using the rest, that also works for me.

The goal is to get a PIPELINE, like how we have in final_submissions/, which can generate the best possible one. The final py file I need. If the final_submissions/ is already the best and there is no improvement, just move those files to the new subfolder I asked and I will see.

So do a mixed strategy, belnds, compound pipeline like I asked. COnsolidate everything and then tell me mathematical best.

3. Similar to final_submissions/ I need a README.md and a report. This is an architecture document so you can tell that we took train.csv, and then split it into 7 pipelines, and then what we did for each etc. So go as in depth as possible, even mentioning the hyperparameters and stuff whatever we have (Their values) so that I can clearly explain everything to the judges. This is imperative and required in depth.
4. Create a weights file. I will pass in a row like in test.csv (With the SMILE and the target) and it should just load from that and infer instead of training the whole thing again. Also need an inference.py so that I can pass in the SMILE and target and it will run it on the model and give the result.

Do this much and give it to me

# Codebase - Explainability Addition
Read AGENTS.md and CONTEXT.md for requrirements. In this folder, there is a new folder called CODEBASES/. In that, the final pipeline is there (Whatever we have done so far)

Take a git commit of everything here before starting.

Goal : Finalize code and score

Let us target the requirements for the Round 3 : Explainability and Interpretable, Robustness against Polymer Invariances (Different Structure), Generalization

There is a Phase_4 on the same path in the GPU laptop (~/Desktop/r3_runtime/Phase_4_Explainability) but also you can write your own scripts. Check it out and then update the CODEBASES/ with this (The main pipeline). Do no reduce the score (Or mariginal reduction only)

Create the final, submission ready py file using these so that I can present it. You need to make sure it is something I can demoonstrate as well (Or a part of it). I will tell you what Im thinking later, but for now ensure that the things for Round 3 are met and we can present it perfectly.

Anything more you can think about, you can add. The purpse of this contest is to submit a perfect pipeline that is research ready and we can show and prove everything that we are saying with evidence. That is the goal. So make the pipeline as I requested. We can even do train test split and stuff to show here and justify them (We cannot use oracle/ anywhere)

Rough Notes : 
+ Think of what to Analyze here
+ Invariance to structures : Data Augmentation will help?
+ Different Polymer Structures but same : Can you show it? Can the Model Find a Relation

# Codebase Refactoring and Addition of EDA etc to the files
Read the below. I ONLY want the full and comprehensive PLAN.md file (However long it may be) and another agent will come adn do the coding, testing, moving and stuff like that. Any research you can do as well, but DO NOT use subagents anywhere. Only YOU can do the searches and stuff. If you want some research docs like about polymers and experiment result analysis reports, you can make and keep (Update those details in PLAN.md so the other agent does not repeat the same task) and then keep it. But dont move anything or modify exist files for now as I have other things running. I will give the next agent PLAN.md and it should do everything else I asked for fro below (Maybe apart from docs which you can make some of) so be as through and details and mention all my requirements, boundaries etc. 

I have a MASSIVE task for you, and you need to do this for me and give it to me. Read AGENTS.md for the context on everything done so far, and take a git commit of everything as well.

Your task for everything below is to create a comprehensive PROMPT.md file. You need to analyze the requriements pasted below, and come up with a prompt, that I will give to another agent to execute. When I say this, I mean that you ahve to go in depth. For example, if I ask for code refactoring or changes, you need to mention where. If I ask for websearch, you need to mention search what. If I asked for docs, you need to mention what all needs to be there in each. The other agent will just read and execute, it it upto you to be through on this. We also need to create new AGENTS.md and CONTEXT.md etc with everything I have pasted below as well.

So far, we have completed the final codebase with scores and the Round 3 requirements (The robustness and Generalization Part).

Now, it is time for us t oconsolidate the codebases and clean and keep it. For this, I have kept a new folder on the desktop called "AISEHack 2.0 Polymer Property Prediction Round 3"

In this, there are three subfolders as below : 
1. Personal : With all docs, files, rough notes for presentation, agents etc. This is the one that I will be using to take my notes, track logs, create docs and presentations etc
2. AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase : This will be the clean codebase for submission
3. Consolidation : It is a consolidation of the files and paths that we have used so far. It is like a cleanup or sorts

All three of them have to have their own git. And the consolidation one has to have more gits underneath it also. You can also ahve a global AGENTS.md and other context files for instructions as needed.

Locations of Analysis and Consolidation
    + Mac:
        + /Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3 : Where the pahses of experiments are there, final CODEBASE/ so far (Before the below required modifications) and oracle etc are present. Context files and stuff are here which you can modify and keep in the required locations.
        + /Users/daver/Desktop/AISEHack-2.0 : Some rough files I kept
    + GPU
        + 4 folders in Desktop/ (r3_runtime, AISEHack-2nd-Edition-Codebase for codebase till phase 2, AISEHack-2.0 and ppp-round-2 shortcut)

Goal : Consolidate paths, context files creation etc for future agents to create codebase, docs, reports, presentations etc

---
---
---
Now, I am going to mention what I need in the different folders. I need you to make a plan, and then tackle it one by one. Here is what all I need in each

#@ Personal
This is where I will operate to do experiments, reports, presentations and QnA

+ FINDINGS.md : Same as Submitted codebase. Something interesting I can show and tell them, like high variance, Tg having a lot of rows and Ei having few (We can tie thos to the architecture also and tell that thats why we chose this one for Tg and this one for Ei etc)
+ STORY.md : This is the story I will be telling in the presentation. We have about 5-6 minutes for this
    + This is how Im imagining my presentation : We start with the problem and the areas where research is lacking (As long as its relevant to our solution), then explain the experiments (Very quickly), then show the eda that we did and best findings (Ones that inspired our pipeline), then show the architecture and walk through the pipeline, then show the explainability robustness generalization part (Have to figure out how to show) and then show a website where people can input the SMILES and target and we make a prediction on it, Then show the leaderboard scores, then mention some future scope like analysis with pretrained models and stuff, then conclude with the same (With links to reports, github repo etc over there).
    + You can see and refine the flow, the story etc. We also need to produce a presentation and report. This is why I need all the consolidated docs and trials so that I can generate these things easily.
+ docs/ : A comprehensive structure and analysis of everything done so far. All experiments, what did and didnt work, why we did something, reasoning, final architecture, locations of files and charts etc etc. I also need comprehensive analysis of the task, the dataset (EDA), what are polymers, what are the research gaps in this task, what are the different properties we are predicting adn their relation etc. 
    + The goal of this folder is so that I can be completely prepared for anything. Like if I need to update my presentation, or if I need to prepare for QnA etc, then I will ask agents and refernece these documents to create. The goal is to be prepared for abslutely anything, be it inside our experiment scope or outside of it
    + THIS IS HIGHLY, HIGHLY COMPREHENSIVE. It should cover all things we have tried in Rounds 2 and 3, web searches for other things like about polymers and property analysis and you need to go as in depth as possible in all of these.
    + This shold cover everything we have done so far, as well as comprehensive QnA as well. So if they as kwhy I think a method did not work, what is the relationshiip between Tg and what it is, why variance was high, Reasoning behind architecture and score (Like why something didnt work and why something did) etc I need to tell them. This requires analysis of the results as well as some websearches.
    + Be prepared to generate reports, presentation slides, demo scripts, QnA etc. As in detail as possible
    + Even EDA and Post inference analysis, findings etc have to be written here so I can pick what I want to weave into my story. Have these in the FINDINGS.md in the codebase for submission also. ANything interesting apart from contest also can be written here. Bias, Calibration, Outliers if any, variances etc as much absed on the domain can be shown here
+ REMAINING_EXPERIMENTS.md : Any remaining experiments that we are yet to try and stuff like that can go here.
    + You dont have to do it, just mentioned the areas of improvement analysis (Weak Targets, Large Datasets etc)
+ TRIALS.md : The list of trials that we did which showed promise / an extensive list if possible. Basically we need to organize by domains (Domain Knowledge based, ml or achtiecture based, hyperparameter based etc as many are needed) and show some experimetns and reasoning as to why they performed well or underperformed etc. Have the ones we are showing in other places (Codebase and stuff) on top and the extensive ist with reference links below that
+ Research/ : Research Papers and Articles for everything, why something did and didnt work
    + They are asking why we choose certain architecture and stuff. We need to backup and validate our claims with research papers wherever possible. GO throug hextensive papers and give me a list so that I can attribute them in my report and presentation
+ Midnight_Report/ : They will ask us to produce a midnight report. There are a few things I need in this
    + There is a folder here /Users/daver/Desktop/AISEHack 2.0 Polymer Property Prediction Round 3/Personal/Sample Reports/ . There are some reports here, and you need to analyze it in DEPTH and make an PROMPT.md file. When needed, I will execute the PROMPT.md file and I expect the ensitre report to be generated. You need to move the folder to this Midnight_Report/ directory and keep it, and ensure you are following the same format and instructions (Analyze them and come up with the best ones. Its ok if the report is over 10 pages long also. Make a 10 page and a 3 page version prompt. Appendix does not count in these). Same as them, we need references, architecture etc.
    + Ideally, everything that is required in teh report should be there in docs/ also. If not, then you need to update the docs/ as well. Thats the point. The PROMPT.md will reference files in other folders like docs/ and the architecture and then make the best possible final report for these. Make sure this is the case and then give me the comprehensive prompt.
    + I am thinking of two appendixes :
        + Experiment Logs : Domains we try (Physics Informed, domain knowlesge based etc) - Keep these consisten throughout all places so that I dont mess things up later when I present.
        + Max cap on score we can get (In Phase 2, I did an oracle assisted heuristic search. If we can somehow mathematically prove that there is a cap like 0.9448 or something that we can achieve based on the information given to us, we can add it to the report)
+ Presentation/
    + Similar to the Report, we need agent skills for presentation here. Mention the requirements, the task, contents etc and keep
    + THere are some sample presentations in this path : /Users/daver/Desktop/AISEHack 2.0 Polymer Property Prediction Round 3/Personal/Sample Reports/ but if you cannot see images dont bother trying to convert or scan it. In the same path, I have made a folder called contents.md where I have the required information as well (Just copied from the presentations)
+ AGENTS.md : Agent instructions for everything that is there above (Where each file is and where what is there etc). I need this to be given so that I can ask agent any question (Like prepare a presentation) and it can reference any docs needed and do it. It can also reference the codebase that is there outside (Submission one) because I dont want to have any agent files there (That should be kept clean for submission ideally)
+ CONTEXT.md : Context of the hackathon as needed
+ Research_Paper/ : We can consolidate the paper from previous round and put it here as well.

I also already have a bunch of files for things like judging criteria, strategy etc. So dont touch them, let them be as it is. I'll move it to Personal/Obsidian/ so you dont topuch them

#@ Submission Codebase
Read CODEBASE/ first, already a lot of things are there over here.

+ Website/ - The website I am going to use in the demo. What I am thinking of is a website so people can input SMILES structures and then 
+ README.md : Contains the structures and quickstart guides. Architecture, individual and final scores etc have to be there here
+ Sandman_Polymer_Property_Prediction.ipynb : You can keep this as a py file, but there are a few additions to do
    + Additions and Modifications to the Codebase : EDA has to be added, like whatever was there in score_descripancy/ (But be smart here, we dont want to show that we are trying to game the system. Little complex EDA to show our findings and stuff we can show here), Charts (We need a MASSIVE amount of charts to show the loss curves, accuracy curves, learnings, explainability, robustness testing etc. Check out some NeurIPS submitted papers and codebased from Kaggle and find interesting charts and resutls to show them. We also cannot use oracle here, so here is what im thinking : Run the full train.csv to get the model weights. Then, in the codebase, do a train-test split to get the metrics to show them and charts whatever is required. Then have a commented out code to do the full train.csv training. Our submission should be the full best model and csv file that I submitted), and finally the post training analysis and inference (Liek we saw some interesting findings of Tg having high variance etc. Again, we cannot use oracle here, have to show with train-test split only)

    + The additions are detailed above. You need to setup the py file like an ipynb file itself. WHen you are done, I will scaffold the py file into an ipynb file and then run everything one by one. Everything required has to be met, including eda, charts etc. USe # for comments and headers, and when I am doing the ipynb, I will need markdown blocks as well for things like architecture, stage, which pipelin etc so let all of those also be in this py file. I will basically just copy it into blocks of code and markdown to make it as an ipynb and then run it. THe submission one will have the charts and stuff displayed so write it in that format. Lastly, ensure that it is easy for me to run, like it automatically activates the venv or has instructions to do it etc. I need to show them the output charts and results and stuff from this, so make it that way.
+ submission.csv : The Submission File that I submitted
+ Optimized_Codes/ : Keep it empty for now, if needed I will tell you later.
+ Experiment_Logs/ : Few of the logs and runs from different phases to be clean and placed here properly (Based on the TRIALS.md in personal. Like we can show them something worked, something did not, something that came close etc and organized by type like domain knowledge based experiments and some architecture based etc. Do not show more than 80 experiments anywhere in codebase that I am submitting. It can be in personal if needed so that in future, I dont retry the experiments and stuff)
+ model.pt : For future inferences (Can be pt or anything else like onnx also, you decide whats best for the situation. It can also be .py if we need because we are splitting differnet pipelines right)
+ inference.py : Like test.csv, people can give a SMILES and a target, and it shuold load model.pt and quickly do inference and give
+ FINDINGS.md : Interesting Findings (test smiles are novel, post analysis shows Tg has high variance, number of entries for Tg and E1 are very disproportional etc. Anything else interesting to tell them in EDA analysis and Post Analysis)
+ ARCHITECTURE.md / METHODOLOGY.md : The end to end architecture of the pipeline, include scores and explainability everywhere (Based on the train test split only)

Approximate README (You can add more or change the order. Just dont make it too bloated, and you can move some things to some docs. So whow some charts, interesting fundings etc and then leave it for the rest later) : 
    Project Overview
    Problem Statement
    Methodology
    Model Architecture
    Key Features
    Dataset
    Results
    Requirements
    Usage
    Future Scope
    Project Structure

    Rough Notes : Loss functions, hyperparmeters, augmentations and if anything is there etc tell all of this in details. How yo usplit the data, how you worked with it etc like an actual ML scienctist. Explain dataset details, models, how you are taking in teh data and preprocessing and infernece etc etc in detial. Scores, results, training, acknowledgements etc

Sample README : https://github.com/Vetri-78640/AISEHack-2026

#@ Consolidation
This folder has the codebases of everything else done so far. We have two laptops and multiple folders, so I want to move and link everything here.

Have an AGENTS.md file here for mine and your reference. 

Do NOT remove the git that is there in the individual folders. They have some tags and stuff in them. Only purpose is to move them together here for better analysis and understanding. 

For the GPU laptop and Round 2 sessions, I dont need them. You just need to maintain a refernece path and instructions to connect to it (Just the path) becasue it is quite long. Only some instreseting finding from experiments, and logs etc can be moved here. Update the TRIALS.md in my personal also so that I can see it and include it in my presentation.

Whatever folders are there in my Mac HAVE to go insto this. Just GPU youdont have to copy, only results are enough.