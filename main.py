import random
import csv
import uuid
import matplotlib.pyplot as plt

from fpdf import FPDF
from database import initialize_db, insert_session
from model import predict_recommendation

quiz = {
  "aiml": [
    {
      "question":
      "Among the following option identify the one which is not a type of learning",
      "options": [
        "Semi supervised learning", "Supervised learning",
        "Reinforcement learning", "Unsupervised learning"
      ],
      "answer":
      "Semi supervised learning"
    },
    {
      "question":
      "Identify the kind of learning algorithm for “facial identities for facial expressions",
      "options": [
        "Prediction", "Recognition Patterns ", "Recognizing anomalies",
        "Generating patterns "
      ],
      "answer":
      "Recognition Patterns "
    },
    {
      "question":
      "Which of the following are common classes of problems in machine learning",
      "options":
      ["Regression", "Classification ", "Clustering", "All of the above"],
      "answer":
      "All of the above"
    },
    {
      "question":
      "Full form of PAC is _____________",
      "options": [
        "Probably Approx Cost", "Probably Approximate Correct ",
        "Probably Approx Communication", "Probably Approximate Computation"
      ],
      "answer":
      "Probably Approximate Correct"
    },
    {
      "question":
      "The total types of the layer in radial basis function neural networks is ______",
      "options": ["1", "2", "3", "4"],
      "answer": "3"
    },
    {
      "question":
      "Which of the following is not machine learning disciplines?",
      "options": [
        "Information Technology", "Optimisation + control", "Physics",
        "Neuro statistics"
      ],
      "answer":
      "Neuro statistics"
    },
    {
      "question":
      "What does K stand for in K mean algorithm?",
      "options": [
        "Number of clusters", "Number of data", "Number of attributes",
        "Number of iterations"
      ],
      "answer":
      "Number of iterations"
    },
    {
      "question":
      "Among the following option identify the one which is used to create the most common graph types.",
      "options": ["plot", "quickplot", "qplot", "All of the above"],
      "answer": "qplot"
    },
    {
      "question":
      "Which of the following is not machine learning?",
      "options": [
        "Artificial Intelligence", "Rule based inference ", "Both A and B",
        "None of the above"
      ],
      "answer":
      "Rule based inference "
    },
    {
      "question":
      "What is the application of machine learning methods to a large database called?",
      "options": [
        "Big data computing", "Internet of things", "Data mining ",
        "Artificial Intelligence"
      ],
      "answer":
      "Data mining "
    },
  ],
  "aids": [
    {
      "question":
      "In how many category processes is Artificial Intelligence classified in?",
      "options": ["Depends on input nature", "5", "2", "3"],
      "answer": "3"
    },
    {
      "question":
      "Which of the following is the common language for Artificial Intelligence",
      "options": ["Python", "Java", "PHP", "Lisp"],
      "answer": "Python"
    },
    {
      "question":
      "Which of the following is not an application of artificial intelligence?",
      "options": [
        "Computer Vision", "Natural Language Processing", "Digital Assistants",
        " Database Management System"
      ],
      "answer":
      " Database Management System"
    },
    {
      "question":
      "Decisions of Victory/Defeat are made in Game trees using which algorithm?",
      "options": ["DFS", "BFS", "Heuristic Search", "Min/ Max Algorithm"],
      "answer": "Min/ Max Algorithm"
    },
    {
      "question": "How many types of observing environments are there?",
      "options": ["2", "3", "1", "0"],
      "answer": "2"
    },
    {
      "question": "How many types of quantification are there in AI?",
      "options": ["1", "2", "3", "4"],
      "answer": "2"
    },
    {
      "question":
      "Machines that try to imitate human intuition while handling vague information lie in the field of AI called?",
      "options":
      ["Functional Logic", "Fuzzy Logic", "Boolean Logic", "Human Logic"],
      "answer":
      "Fuzzy Logic"
    },
    {
      "question":
      "Which of the following are valid Machine Learning algorithms?",
      "options": [
        "Linear Regression", "K Means Clustering", "Naive Bayes",
        "All of the above"
      ],
      "answer":
      "All of the above"
    },
    {
      "question":
      "How is a decision reached upon by a decision tree?",
      "options":
      ["No test", "Single Test", "Double Test", "Multiple sequences of tests"],
      "answer":
      "Multiple sequences of tests"
    },
    {
      "question":
      "How is the brightness of a pixel on the screen increased?",
      "options": [
        "Increasing the amount of light directed on the screen",
        "Using mirrors", "Using sound waves", "None of the above"
      ],
      "answer":
      "Increasing the amount of light directed on the screen"
    },
  ],
  "cyber": [
    {
      "question":
      "Identify the term which denotes that only authorized users are capable of accessing the information",
      "options":
      ["Confidentiality", "Availability", "Integrity", "Non-repudiation"],
      "answer":
      "Availability"
    },
    {
      "question":
      "Identify the oldest phone hacking technique used by hackers to make free calls",
      "options": ["Spamming", "Phreaking", "Cracking", "Phishing"],
      "answer": "Phreaking"
    },
    {
      "question":
      "Identify among the following which is used to avoid browser-based hacking",
      "options": [
        "Adware remover in browser", "Incognito mode in the browser",
        "Anti-malware in browser", "Remote browser access"
      ],
      "answer":
      "Remote browser access"
    },
    {
      "question":
      "Which of the following platforms is used for the safety and protection of information in the cloud?",
      "options": [
        "AWS", "Cloud workload protection platforms",
        "Cloud security protocols", "One Drive"
      ],
      "answer":
      "Cloud workload protection platforms"
    },
    {
      "question":
      " Choose among the following techniques, which are used to hide information inside a picture",
      "options":
      ["Image rendering", "Steganography", "Rootkits", "Bitmapping"],
      "answer": "Steganography"
    },
    {
      "question":
      "EDR stands for _____?",
      "options": [
        "Endless detection and response", "Endpoint detection and response ",
        "Endless detection and recovery", "Endpoint detection and recovery"
      ],
      "answer":
      "Endpoint detection and response "
    },
    {
      "question":
      "Which of the following is used for monitoring traffic and analyzing network flow?",
      "options": [
        "Managed detection and response", "Cloud access security broker",
        "Network traffic analysis ", "Network security firewall"
      ],
      "answer":
      "Network traffic analysis "
    },
    {
      "question":
      "In which category does compromising confidential information fall?",
      "options": ["Threat ", "Bug", "Attack", "Vulnerability"],
      "answer": "Threat "
    },
    {
      "question": "Identify the class of computer threats.",
      "options": ["Phishing", "DOS attack ", "Soliciting", "Both B and C"],
      "answer": "DOS attack "
    },
    {
      "question":
      "Which software is mainly used to help users detect viruses and avoid them?",
      "options": ["Antivirus", "Adware", "Malware", "None of the above"],
      "answer": "Antivirus"
    },
  ],
  "full-stack": [
    {
      "question":
      "Among the following operators identify the one which is used to allocate memory to array variables in JavaScript.",
      "options": ["New", "new malloc", "alloc", "malloc"],
      "answer": "New"
    },
    {
      "question":
      "Why were cookies designed?",
      "options": [
        "for server-side programming", "For client-side programming",
        "both a and b", "None"
      ],
      "answer":
      "for server-side programming"
    },
    {
      "question":
      "What are variables used in JavaScript programs",
      "options": [
        "Varying randomly", "Storing numbers, dates, and other values ",
        "Used as header files", "none of the above"
      ],
      "answer":
      "Storing numbers, dates, and other values "
    },
    {
      "question":
      "Simple network management protocol uses which of the following port number?",
      "options": ["164", "163", "160", "161"],
      "answer": "161"
    },
    {
      "question":
      "Full form of W3C is _____________",
      "options": [
        "World Wide Websites community", "World Wide Web community",
        "World Wide Websites consortium ", "World Wide Web consortium"
      ],
      "answer":
      "World Wide Websites consortium "
    },
    {
      "question":
      "When is the body tag used?",
      "options":
      ["after FORM tag", "after Title tag", "after EM tag", "after HEAD tag "],
      "answer":
      "after HEAD tag  "
    },
    {
      "question":
      "Which of the following attribute defines the relationship between the current document and URL?",
      "options": ["URL", "REV", "REL ", "None"],
      "answer": "REL "
    },
    {
      "question":
      "Why is XPATH used?",
      "options": [
        "To address the server ", "to store the IP address of this server",
        "to address the document by specifying a location path ", "None"
      ],
      "answer":
      "to address the document by specifying a location path  "
    },
    {
      "question":
      "What is the effect of the \<b> tag?",
      "options": [
        "It converts the text within it to a bold tag.",
        "It is used to write black-coloured font.",
        "It is used to change the font size.", "None"
      ],
      "answer":
      "It converts the text within it to a bold tag. "
    },
    {
      "question":
      "What is the function of the HTML style attribute?",
      "options": [
        "It is used to add styles to an HTML element. ",
        "It is used to uniquely identify some specific styles of some elements.",
        "Both (a) and (b)", "None"
      ],
      "answer":
      "It is used to add styles to an HTML element. "
    },
  ],
  "app": [
    {
      "question":
      "What is an activity in android?",
      "options": [
        "android class", "android package",
        "A single screen in an application with supporting java code",
        "None of the above"
      ],
      "answer":
      "A single screen in an application with supporting java code"
    },
    {
      "question":
      "Among the following options choose the one for which Android is based on   Linux.",
      "options":
      ["Networking", "Portability", "Security", "All of the above "],
      "answer": "All of the above "
    },
    {
      "question":
      "Among the below virtual machines choose the one which is used by the Android operating system",
      "options":
      ["Dalvik operating system ", "JVM", "Simple virtual machine", "none"],
      "answer":
      "Dalvik operating system "
    },
    {
      "question":
      "What is manifest XML in android?",
      "options": [
        "It has information about layout in an application",
        "It has all the information about an application ",
        "It has the information about activities in an application", "none"
      ],
      "answer":
      "It has all the information about an application "
    },
    {
      "question":
      "What is the use of a content provider in Android?",
      "options": [
        "For sharing the data between applications",
        "For storing the data in the database",
        "For sending the data from an application to another application",
        "None of the above"
      ],
      "answer":
      "For sending the data from an application to another application "
    },
    {
      "question":
      "While developing android application developers can test their apps on",
      "options": [
        "Emulators in Android SDK", "Android Phone", "Third-Party Emulator",
        "All the above "
      ],
      "answer":
      "All the above"
    },
    {
      "question":
      "The full form of ADB is.",
      "options": [
        "Android Delete Bridge", "Android Debug Bridge ",
        "Android Destroy Bridge", "Android Developed Bridge"
      ],
      "answer":
      "Android Debug Bridge "
    },
    {
      "question": "Choose the option which is contained in the src folder",
      "options": ["Manifest", "Java Source Code ", "XML", "All of them"],
      "answer": "Java Source Code "
    },
    {
      "question":
      "Identify the lowest layer of Android architecture.",
      "options": [
        "Applications", "Applications framework", "Linux Kernel ",
        "System libraries and android runtime"
      ],
      "answer":
      "Linux Kernel"
    },
    {
      "question": "Choose the built-in database of Android.",
      "options": ["MySQL", "SQLite ", "Oracle", "None"],
      "answer": "SQLite "
    },
  ]
}

# Ensure DB is ready before quiz starts
initialize_db()

def administer_quiz():
  name = input("What is your name? ")
  email = input("What is your email address? ")
  score = {}
  for topic in quiz:
    score[topic] = 0
    print(f"{topic.capitalize()} quiz:")
    for question in quiz[topic]:
      print(question["question"])
      random.shuffle(question["options"])
      for i, option in enumerate(question["options"]):
        print(f"{i+1}. {option}")
      answer = input("Your answer (enter the option number): ")
      if question["options"][int(answer) - 1] == question["answer"]:
        score[topic] += 1
        print("Correct!")
      else:
        print("Incorrect.")
    print(f"{topic.capitalize()} quiz score: {score[topic]}\n")
  print("Quiz complete!")
  for topic in score:
    print(f"{topic.capitalize()} quiz score: {score[topic]}")

  with open("quiz_scores.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Topic", "Score"])
    for topic, s in score.items():
      writer.writerow([topic, s])

  # ── Save to SQLite database ──
  session_id = str(uuid.uuid4())
  insert_session(session_id, name, email, score)
  print(f"\n[DB] Results saved to database (session: {session_id[:8]}...)")

  # ── Generate and SAVE the chart FIRST (before PDF embeds it) ──
  topics_list = list(score.keys())
  scores_list = list(score.values())

  colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
  plt.figure(figsize=(9, 5))
  bars = plt.bar(topics_list, scores_list, color=colors, edgecolor="white", linewidth=0.8)
  plt.title(f"Quiz Score Report — {name}", fontsize=14, fontweight="bold", pad=15)
  plt.xlabel("Topic", fontsize=11)
  plt.ylabel("Score (out of 10)", fontsize=11)
  plt.ylim(0, 10)
  plt.xticks(fontsize=10)

  # Label each bar with its score
  for bar, s in zip(bars, scores_list):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
             str(s), ha="center", va="bottom", fontsize=11, fontweight="bold")

  # ── ML-powered recommendation ──
  ml_recommendation = predict_recommendation(score)
  print(f"\n🤖 AI Recommendation (ML Model v1.2): {ml_recommendation.upper()}")

  # Highlight the best topic
  max_s = max(scores_list)
  max_topic = ml_recommendation  # use ML model output
  plt.figtext(0.5, 0.01,
              f"✅ Recommended Course: {max_topic.upper()} (your strongest area — score: {max_s}/10)",
              ha="center", fontsize=10, color="#333333",
              bbox=dict(facecolor="#e8f5e9", edgecolor="#4caf50", boxstyle="round,pad=0.4"))

  plt.tight_layout(rect=[0, 0.06, 1, 1])
  plt.savefig("quiz_scores.png", dpi=150)
  plt.show()

  print(f"\n📊 Chart saved → quiz_scores.png")

  # ── Build the PDF report (after chart is saved) ──
  from fpdf import FPDF, XPos, YPos
  pdf = FPDF()
  pdf.add_page()

  # Header
  pdf.set_fill_color(41, 98, 155)
  pdf.rect(0, 0, 210, 28, "F")
  pdf.set_text_color(255, 255, 255)
  pdf.set_font("Helvetica", "B", 18)
  pdf.set_y(8)
  pdf.cell(0, 12, "Course Coach — Quiz Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

  # Student info
  pdf.set_text_color(30, 30, 30)
  pdf.set_fill_color(240, 245, 255)
  pdf.rect(10, 32, 190, 22, "F")
  pdf.set_font("Helvetica", "", 11)
  pdf.set_y(35)
  pdf.cell(0, 7, f"  Name:  {name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
  pdf.cell(0, 7, f"  Email: {email}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

  # Scores table
  pdf.set_y(60)
  pdf.set_font("Helvetica", "B", 12)
  pdf.cell(0, 8, "Topic Scores", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
  pdf.set_font("Helvetica", "", 11)
  for t, s in score.items():
    pct = int((s / 10) * 100)
    pdf.cell(0, 7, f"  {t.capitalize():<15}  {s}/10   ({pct}%)",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

  # Recommendation
  pdf.set_y(140)
  pdf.set_fill_color(232, 245, 233)
  pdf.set_draw_color(76, 175, 80)
  pdf.rect(10, pdf.get_y(), 190, 14, "DF")
  pdf.set_font("Helvetica", "B", 11)
  pdf.cell(0, 14, f"  ✅ Recommended Course: {max_topic.upper()} (Score: {max_s}/10)",
           new_x=XPos.LMARGIN, new_y=YPos.NEXT)

  # Embed chart
  pdf.image("quiz_scores.png", x=10, y=160, w=185)

  pdf_name = f"{name.replace(' ', '_')}_quiz_report.pdf"
  pdf.output(pdf_name)
  print(f"📄 PDF report saved → {pdf_name}")

  return score


score = administer_quiz()
