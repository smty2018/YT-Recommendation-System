import csv

data = [
    ["Data", "Label"],
    ["Cat videos", "Not Related"],
    ["Social media", "Not Related"],
    ["Online shopping", "Not Related"],
    ["Memes", "Not Related"],
    ["Video games", "Not Related"],
    ["Celebrity gossip", "Not Related"],
    ["Reality TV", "Not Related"],
    ["Cooking shows", "Not Related"],
    ["DIY crafts", "Not Related"],
    ["Fashion trends", "Not Related"],
    ["Makeup tutorials", "Not Related"],
    ["Sports highlights", "Not Related"],
    ["Streaming music", "Not Related"],
    ["Podcasts", "Not Related"],
    ["Stand-up comedy", "Not Related"],
    ["Movie trailers", "Not Related"],
    ["Travel vlogs", "Not Related"],
    ["Fitness workouts", "Not Related"],
    ["Home decor", "Not Related"],
    ["Gardening tips", "Not Related"],
    ["Conspiracy theories", "Not Related"],
    ["Urban legends", "Not Related"],
    ["Paranormal activities", "Not Related"],
    ["Cryptocurrency news", "Not Related"],
    ["Stock market updates", "Not Related"],
    ["Weather forecasts", "Not Related"],
    ["Traffic reports", "Not Related"],
    ["Fashion magazines", "Not Related"],
    ["Tabloid headlines", "Not Related"],
    ["Celebrity scandals", "Not Related"],
    ["Product reviews", "Not Related"],
    ["Car racing", "Not Related"],
    ["Extreme sports", "Not Related"],
    ["Wildlife documentaries", "Not Related"],
    ["History documentaries", "Not Related"],
    ["Crime dramas", "Not Related"],
    ["Romance novels", "Not Related"],
    ["Fantasy fiction", "Not Related"],
    ["Science fiction", "Not Related"],
    ["Mystery novels", "Not Related"],
    ["True crime stories", "Not Related"],
    ["Horror movies", "Not Related"],
    ["Action movies", "Not Related"],
    ["Drama series", "Not Related"],
    ["Sitcoms", "Not Related"],
    ["Animation", "Not Related"],
    ["Cooking competitions", "Not Related"],
    ["Food reviews", "Not Related"],
    ["Recipe blogs", "Not Related"],
    ["Craft tutorials", "Not Related"],
    ["Makeup reviews", "Not Related"],
    ["Gaming streams", "Not Related"],
    ["E-sports tournaments", "Not Related"],
    ["Music concerts", "Not Related"],
    ["Dance performances", "Not Related"],
    ["Art exhibitions", "Not Related"],
    ["Photography galleries", "Not Related"],
    ["Fashion shows", "Not Related"],
    ["Travel photography", "Not Related"],
    ["Nature documentaries", "Not Related"],
    ["Space exploration news", "Not Related"],
    ["Wildlife conservation", "Not Related"],
    ["Environmental activism", "Not Related"],
    ["Healthy eating tips", "Not Related"],
    ["Fitness challenges", "Not Related"],
    ["Yoga retreats", "Not Related"],
    ["Meditation guides", "Not Related"],
    ["Self-help books", "Not Related"],
    ["Personal development", "Not Related"],
    ["Leadership seminars", "Not Related"],
    ["Motivational speakers", "Not Related"],
    ["Entrepreneurship podcasts", "Not Related"],
    ["Business news", "Not Related"],
    ["Investment tips", "Not Related"],
    ["Financial planning", "Not Related"],
    ["Career advice", "Not Related"],
    ["Job search tips", "Not Related"],
    ["Educational apps", "Not Related"],
    ["Online courses", "Not Related"],
    ["Language learning", "Not Related"],
    ["Study tips", "Not Related"],
    ["Academic research", "Not Related"],
    ["Book clubs", "Not Related"],
    ["Writing workshops", "Not Related"],
    ["Public speaking courses", "Not Related"],
    ["Debate clubs", "Not Related"],
    ["Scientific research", "Not Related"],
    ["Technology news", "Not Related"],
    ["Innovation trends", "Not Related"],
    ["Robotics competitions", "Not Related"],
    ["Coding challenges", "Not Related"],
    ["Game development forums", "Not Related"],
    ["Web design tutorials", "Not Related"],
    ["Software updates", "Not Related"],
    ["Tech gadgets reviews", "Not Related"],
    ["IT support forums", "Not Related"],
    ["Cybersecurity tips", "Not Related"],
    ["Cloud computing news", "Not Related"],
    ["Data privacy concerns", "Not Related"],
    ["Internet culture", "Not Related"],
    ["Vector database", "Related to Education Study Coding"],
    ["PyTorch", "Related to Education Study Coding"],
    ["Matrix decomposition", "Related to Education Study Coding"],
    ["Machine learning algorithms", "Related to Education Study Coding"],
    ["Data structures", "Related to Education Study Coding"],
    ["Neural networks", "Related to Education Study Coding"],
    ["Computer vision", "Related to Education Study Coding"],
    ["Algorithm complexity", "Related to Education Study Coding"],
    ["Reinforcement learning", "Related to Education Study Coding"],
    ["Natural language processing", "Related to Education Study Coding"],
    ["Deep learning frameworks", "Related to Education Study Coding"],
    ["Compiler optimization", "Related to Education Study Coding"],
    ["Parallel computing", "Related to Education Study Coding"],
    ["Big-O notation", "Related to Education Study Coding"],
    ["CUDA programming", "Related to Education Study Coding"],
    ["Robotics algorithms", "Related to Education Study Coding"],
    ["Game theory", "Related to Education Study Coding"],
    ["Cloud computing architectures", "Related to Education Study Coding"],
    ["Cybersecurity protocols", "Related to Education Study Coding"],
    ["Blockchain technology", "Related to Education Study Coding"],
    ["Cryptography techniques", "Related to Education Study Coding"],
    ["Internet of Things (IoT) protocols", "Related to Education Study Coding"],
    ["Data mining methods", "Related to Education Study Coding"],
    ["Algorithm design", "Related to Education Study Coding"],
    ["Software engineering principles", "Related to Education Study Coding"],
    ["Operating systems design", "Related to Education Study Coding"],
    ["Network protocols", "Related to Education Study Coding"],
    ["Database management systems", "Related to Education Study Coding"],
    ["Web development frameworks", "Related to Education Study Coding"],
    ["Programming languages theory", "Related to Education Study Coding"],
    ["Distributed computing models", "Related to Education Study Coding"],
    ["High-performance computing", "Related to Education Study Coding"],
    ["Quantum computing algorithms", "Related to Education Study Coding"],
    ["Bioinformatics algorithms", "Related to Education Study Coding"],
    ["Computational geometry", "Related to Education Study Coding"],
    ["Evolutionary algorithms", "Related to Education Study Coding"],
    ["Statistical learning theory", "Related to Education Study Coding"],
    ["Information retrieval systems", "Related to Education Study Coding"],
    ["Semantic web technologies", "Related to Education Study Coding"],
    ["Wireless communication protocols", "Related to Education Study Coding"],
    ["Digital signal processing", "Related to Education Study Coding"],
    ["Image processing techniques", "Related to Education Study Coding"],
    ["Augmented reality development", "Related to Education Study Coding"],
    ["Virtual reality systems", "Related to Education Study Coding"],
    ["Mobile application development", "Related to Education Study Coding"],
    ["Data visualization techniques", "Related to Education Study Coding"],
    ["Graph theory applications", "Related to Education Study Coding"],
    ["Mathematical optimization methods", "Related to Education Study Coding"],
    ["Embedded systems programming", "Related to Education Study Coding"],
    ["Compiler construction", "Related to Education Study Coding"],
    ["Game development frameworks", "Related to Education Study Coding"],
    ["Functional programming concepts", "Related to Education Study Coding"],
    ["Software testing methodologies", "Related to Education Study Coding"],
    ["DevOps practices", "Related to Education Study Coding"],
    ["Version control systems", "Related to Education Study Coding"],
    ["Agile software development", "Related to Education Study Coding"],
    ["UX/UI design principles", "Related to Education Study Coding"],
    ["Human-computer interaction", "Related to Education Study Coding"],
    ["Software architecture patterns", "Related to Education Study Coding"],
    ["Data compression algorithms", "Related to Education Study Coding"],
    ["Computational linguistics", "Related to Education Study Coding"],
    ["Network security protocols", "Related to Education Study Coding"],
    ["Cloud storage solutions", "Related to Education Study Coding"],
    ["Scalable computing architectures", "Related to Education Study Coding"],
    ["Knowledge representation methods", "Related to Education Study Coding"],
    ["Information theory concepts", "Related to Education Study Coding"],
    ["Compiler optimization techniques", "Related to Education Study Coding"],
    ["Robotics programming languages", "Related to Education Study Coding"],
    ["Bioinformatics data analysis", "Related to Education Study Coding"],
    ["Computational neuroscience", "Related to Education Study Coding"],
    ["Quantum information theory", "Related to Education Study Coding"],
    ["Internet protocols", "Related to Education Study Coding"],
    ["Database normalization", "Related to Education Study Coding"],
    ["Web application security", "Related to Education Study Coding"],
    ["Programming language paradigms", "Related to Education Study Coding"],
    ["Distributed systems architecture", "Related to Education Study Coding"],
    ["High-performance algorithms", "Related to Education Study Coding"],
    ["Parallel programming models", "Related to Education Study Coding"],
    ["Embedded software development", "Related to Education Study Coding"],
    ["Computational fluid dynamics", "Related to Education Study Coding"],
    ["Predictive modeling techniques", "Related to Education Study Coding"],
    ["Cyber-physical systems", "Related to Education Study Coding"],
    ["Data-driven decision making", "Related to Education Study Coding"],
    ["Natural computing algorithms", "Related to Education Study Coding"],
    ["Semantic analysis methods", "Related to Education Study Coding"],
    ["Wireless sensor networks", "Related to Education Study Coding"],
    ["Digital forensics methods", "Related to Education Study Coding"],
    ["Compiler theory", "Related to Education Study Coding"],
    ["Game engine design", "Related to Education Study Coding"],
    ["Functional data structures", "Related to Education Study Coding"],
    ["Software refactoring techniques", "Related to Education Study Coding"],
    ["Network simulation tools", "Related to Education Study Coding"],
    ["Cloud-native application development", "Related to Education Study Coding"],
    ["Quantum computing languages", "Related to Education Study Coding"],
    ["Computational biology algorithms", "Related to Education Study Coding"],
    ["Network routing protocols", "Related to Education Study Coding"],
    ["Database query optimization", "Related to Education Study Coding"],
    ["Web framework development", "Related to Education Study Coding"],
    ["Programming language semantics", "Related to Education Study Coding"],
    ["Generative AI", "Related to Education Study Coding"],
    ["Political campaign ads", "Not Related"],
    ["Celebrity gossip news", "Not Related"],
    ["Fashion trends blogs", "Not Related"],
    ["DIY home improvement videos", "Not Related"],
    ["Fitness influencer workouts", "Not Related"],
    ["Video game live streams", "Not Related"],
    ["Travel agency promotions", "Not Related"],
    ["Gardening tutorials", "Not Related"],
    ["Cooking recipe reviews", "Not Related"],
    ["Movie critique blogs", "Not Related"],
    ["open", "Not Related"],
    ["close", "Not Related"],
    ["ll", "Not Related"],
    ["and", "Not Related"],
     ["Natural language understanding", "Related to Education Study Coding"],
    ["Robot perception algorithms", "Related to Education Study Coding"],
    ["Neural network architectures", "Related to Education Study Coding"],
    ["Deep reinforcement learning", "Related to Education Study Coding"],
    ["Graph neural networks", "Related to Education Study Coding"],
    ["Bioinformatics sequence analysis", "Related to Education Study Coding"],
    ["Machine translation models", "Related to Education Study Coding"],
    ["Quantum error correction", "Related to Education Study Coding"],
    ["Blockchain consensus mechanisms", "Related to Education Study Coding"],
    ["Cybersecurity threat detection", "Related to Education Study Coding"]
]


csv_file = "related_topics.csv"


with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"CSV file '{csv_file}' has been successfully created.")
