# Contributing

OT-Agent is a highly collaborative project and we welcome many contributors. Below we have detailed the steps to contribute new code and outlined a few guidelines (marked as **Important**) that will make it easier to all work together. 

## Clone the repo
```
git clone git@github.com:mlfoundations/dc-agent.git
```

## Create a new branch
``` 
git switch -c ryan/contributing-docs
```

> [!IMPORTANT]
> Use the  convention `your-name/branch-name`
> 
> (this way we know who owns what code)

## Create a PR
```
# install with brew install gh (macOS) or sudo apt install gh (Ubuntu)
# login with gh auth login
gh pr create \
  --draft \
  --title "Add contributing.md" \
  --body "Outline the conventions used for contributing code to dc-agent" 
```

> [!IMPORTANT]
> Mark your PR as a draft if it is still in progress

## Request a review

Request a review from the DRI (directly responsible individual) for the area of the project you are contributing to. 
- Data is Etash: `EtashGuha`
- RL is Tyler and Charlie: `tyler-griggs` and `CharlieFRuan`
- SFT is Ben: `penfever`
- Eval is Negin: `neginraoof`

If your code crosses multiple areas, include all relevant DRIs. The reviewers will acknowledge and provide a preliminary comment in 24 hours. We will try to get PRs merged as quickly as possible. Please respond to changes from review quickly and if your reviewer is not responsive, please ping them on Discord sooner rather than later!

```
# switch PR from draft to ready
gh pr ready

# request reviews
gh pr edit --add-reviewer EtashGuha,tyler-griggs,CharlieFRuan,penfever,neginraoof
```

> [!IMPORTANT]
> Make sure your PRs are small and modular so it is easy to review
> 
> (LLM generated PRs can be sprawling and touch more code than necessary, so make sure to adjust / prune those as necessary)