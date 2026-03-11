from transformers import pipeline

# Part 2 - Default Sentiment Analysis
model = pipeline("sentiment-analysis")
text = "This is the most straightforward and effective method I have ever learned"
result = model(text)
print("Part 2:", result)

# Part 3 - Custom Finance Model
financial_classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
text2 = "The stock market rally continued, suggesting strong long-term growth."
result2 = financial_classifier(text2)
print("Part 3:", result2)

# Part 4 - Bulk Analysis
sentences = [
    "The quarterly earnings report was surprisingly weak, causing investor concern.",
    "Despite market volatility, the company announced record profits.",
    "I'm not sure if I should invest in tech stocks this quarter."
]
results4 = financial_classifier(sentences)
for sentence, res in zip(sentences, results4):
    print(f"Text: {sentence}")
    print(f"Result: {res}\n")

# Part 5 - Zero Shot Classification
zsc_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
text5 = "The latest press release details the company's new policy on remote work, including guidelines for team communication and hard deadlines for employees worldwide."
labels = ["Employee Relations", "Financial News", "Product Announcement", "Technical Support", "Sales", "HR Policy", "Legal Compliance"]
result5 = zsc_model(text5, labels)
print("Zero-Shot Classification Results")
print(f"Input Text: {result5['sequence']}")
print("Classification Scores:")
for i, (label, score) in enumerate(zip(result5['labels'], result5['scores'])):
    print(f"  {i+1}. {label}: {round(score * 100, 2)}%")