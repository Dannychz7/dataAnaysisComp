Name: Daniel Chavez

DISCLAIMER: I did create or own the dataComparative.csv file. I got the sample data from: https://github.com/uhh-lt/comparative. They are the rightful owners of the data.

# Basic Git Setup
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
git config --global core.editor "code --wait"

# Cloning a Repository
git clone https://github.com/your-username/your-repo.git  # Clone with HTTPS
git clone git@github.com:your-username/your-repo.git  # Clone with SSH

# Checking Repository Status
git status  # Show changes and untracked files
git log  # View commit history
git show <commit-hash>  # Show details of a specific commit

# Making Changes
git add .  # Stage all modified and new files
git add filename  # Stage a specific file
git commit -m "Your commit message"  # Commit staged changes
git commit --amend -m "Updated commit message"  # Edit last commit message

# Pushing Changes
git push origin main  # Push changes to GitHub (replace `main` with `master` if needed)
git push origin branch-name  # Push a specific branch

# Pulling Changes
git pull origin main  # Fetch and merge the latest changes
git pull --rebase origin main  # Rebase instead of merge

# Branching & Merging
git branch  # List branches
git branch new-branch  # Create a new branch
git checkout new-branch  # Switch to a branch
git switch new-branch  # Alternative way to switch to a branch
git merge branch-name  # Merge a branch into the current branch
git branch -d branch-name  # Delete a branch

# Working with Remote Repositories
git remote -v  # View remote repositories
git remote add origin https://github.com/your-username/your-repo.git  # Add a remote repo
git remote remove origin  # Remove a remote repo

# Undoing Changes
git reset --hard HEAD  # Undo all local changes
git reset HEAD~1  # Undo the last commit (but keep changes)
git checkout -- filename  # Discard changes to a specific file
git revert <commit-hash>  # Revert a specific commit

# Stashing Changes
git stash  # Save changes temporarily
git stash list  # View stashed changes
git stash apply  # Restore stashed changes
git stash drop  # Delete the last stash

# Checking Differences
git diff  # Show unstaged changes
git diff --staged  # Show staged changes

# Deleting Files & Commits
git rm filename  # Delete a file and stage the deletion
git reset --hard <commit-hash>  # Reset to a specific commit and discard all changes



You will probably use these the most:
    - git pull origin main
    - git push origin main
    - git add . 
    - git commit -m "msg"

Example:
    - I made a change to raw_data_Hist.py
    1. git add . # Stage all modified and new files
    2. git commit -m "Add some msg here"
    3. git push origin main #This should upload to the github repo successfully
