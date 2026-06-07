#!/bin/bash
cd "d:\downloads\exams projects 2026\ds project management" || exit 1

# Initialize git
git init

# Configure git user
git config user.email "mariemtnb@example.com"
git config user.name "Mariem Tanabene"

# Add remote
git remote add origin https://github.com/mariemtnb/NLP-Project.git

# Add all files
git add .

# Commit
git commit -m "Initial commit: SentiYelp - NLP Sentiment Analysis with CRISP-DM Methodology

- Dataset: 51,000 balanced Yelp reviews
- Models: LinearSVC (F1=0.743), Naive Bayes, VADER baseline
- Features: Flask API + Web Dashboard
- Documentation: CRISP-DM Report, Agile Retrospective, 3 Jupyter Notebooks
- Tableau: 9 CSV exports for Business Intelligence
- Architecture: Data pipeline, model training, REST API, frontend

Team: Mariem Tanabene, Abir Mhamdi, Nouha Laaroussi"

# Set main branch and push
git branch -M main
git push -u origin main

echo "Upload completed successfully!"
