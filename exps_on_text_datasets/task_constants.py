task_to_benchmark = {
    # GLUE
    "cola": "glue",
    "mnli": "glue",
    "mrpc": "glue",
    "qnli": "glue",
    "qqp":  "glue",
    "sst2": "glue",
    "stsb": "glue",
    "wnli": "glue",
    # SuperGLUE excluding "rte": ("premise", "hypothesis"), since it is the same as in GLUE
    "boolq":   "super_glue",
    "cb":      "super_glue", 
    "wic":     "super_glue", 
    "copa":    "super_glue",
    "multirc": "super_glue",
    "wsc.fixed":     "super_glue",
    "rte":     "super_glue",
    "record":  "super_glue",
    # Summerization,
    "xsum": None,
    "samsum": None,
    "multi_news": None,
    # Generation
    "e2e_nlg_cleaned": None,
    "web_nlg_en": "gem",
    "common_gen": None,
    # Sentiment Classification
    "imdb": None,
    "yelp_review_full": None,
    # open-domain QA
    "social_i_qa": None,
    "wiki_qa": None,
    "fullwiki": "hotpot_qa",
    # Coreference Resolution
    "winogrande_debiased": "winogrande",
    "quoref": None,
}

task_to_null_template = {
    "multirc": '''{{paragraph}} {{question}} {{answer}}
        ||| {% if label != -1 %}{{answer_choices[label]}}{% endif %}''',
    "boolq": '''{{ passage }} {{ question }} 
        ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}''',
    "rte": '''{{premise}} {{hypothesis}}
        ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}''',
    "wic": '''{{sentence1}} {{sentence2}} {{word}}
        ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}''',
    "xsum": '''{{document}}
        ||| {{summary}}''',
    "samsum": '''{{dialogue}}
        ||| {{summary}}''',
    "multi_news": '''{{document}}
        ||| {{summary[2:]}}''',
    "e2e_nlg_cleaned": '''
      {% for feature in meaning_representation.split("]") %} {% set key = feature.split("[")[0].replace(",","")
      %} {% set value = feature.replace(",","").replace(key+"[", '''''''') %}
      {% if value != "" %} {{key}} : {{value}} {% endif %}
      {%- endfor %}
      ||| {{human_reference}}''',
    "web_nlg_en": ''' {{input| join(", ")}} 
        ||| {{target}}''' 
}

task_to_instruction_template = {

    # For GLUE benchmark
    "cola" : '''{{sentence}}

      Is this example grammatically correct and sensible?

      |||

      {{ answer_choices[label] }}''',
    
    "mnli" : '''{{premise}} Based on the previous passage, is it true that "{{hypothesis}}"?
      Yes, no, or maybe? ||| {{ answer_choices[label] }}''',

    "mrpc" : '''Does the sentence

      {{sentence1}}

      paraphrase (that is, mean the same thing as) this sentence?

      {{sentence2}}

      |||

      {{ answer_choices[label] }}''',

    "qnli" : '''Can you answer the question "{{question}}" based only on the following:

      {{sentence}}

      |||

      {{answer_choices[label]}}''',

    "qqp" : '''I received the questions "{{question1}}" and "{{question2}}". Are they
      duplicates? ||| {{ answer_choices[label] }}''',

    "sst2" : '''I''m reading a review that says "{{sentence}}".


      Do you think the review is {{"positive"}} or {{"negative"}}? ||| {{ answer_choices[label]
      }}''',

    "stsb" : '''Please rate how similar these two sentences are from {{"0.0"}} to {{"5.0"}}.

      Sentence A: {{sentence1}}

      Sentence B: {{sentence2}}

      |||

      {{ (((5*label) | round )/5) }}''',

    "wnli" : '''Assume that the following is true:

      {{sentence1}}

      Does this mean that "{{sentence2}}"?

      |||

      {{answer_choices[label]}}''',

    # For superGLUE 

    "boolq" : '''{{ passage }} \n\nHaving read that, I wonder {{ question }}? |||\n{% if
       label != -1 %}\n{{ answer_choices[label] }} \n{% endif %}''',

    "cb" : '''{{premise}} Based on the previous passage, is it true that "{{hypothesis}}"?
      Yes, no, or maybe? ||| {% if label !=-1 %}{{ answer_choices[label] }}{% endif
      %}''',

    "wic" : '''Does the word "{{word}}" have the same meaning in these two sentences?

      {{sentence1}}

      {{sentence2}}

      ||| {% if label != -1%}

      {{answer_choices[label]}}

      {% endif %}''',

    "copa" : '''Pick the more likely continuation to the following sentence:

      {{ premise }} {% if question == "cause" %} as a result of: {% else %} as a consequence:
      {% endif %}

      - {{choice1}}

      - {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}''',

    "multirc" : '''{{paragraph}}


      Question: {{question}}

      I found this answer "{{answer}}". Is that correct? Yes or no?

      |||

      {% if label != -1 %}{{answer_choices[label]}}{% endif %}''',

    "wsc.fixed" : '''{{ text }} In the previous sentence, does the pronoun "{{ span2_text.lower()
      }}" refer to {{ span1_text }}? Yes or no? ||| {% if label != -1 %}{{ answer_choices[label]
      }}{% endif %}''',

    "rte" : '''{{premise}} Using only the above description and what you know about the
      world, is "{{hypothesis}}" definitely correct? Yes or no? ||| {% if label !=
      -1 %}{{ answer_choices[label] }}{% endif %}''',

    "record" : '''Summary:\n\n- {{ passage.split(\"@highlight\")[1:] | join(\"\\n- \") }}
       \n\nArticle:\n\n{{ passage.split(\"@highlight\")[0] }}\n\nNow that you've
       read the article, please write a new sentence to add to it.\n\n||| {% if (
       answers | length ) > 0 %}{{ query | replace(\"@placeholder\", answers | choice)
       }} {% endif %}''',

    # Summerization
    "xsum" : '''{{document}}


      ===


      Write a summary of the text above : ||| {{summary}}''',

    "samsum" : '''Summarize this dialogue: {{dialogue}} |||

      {{summary}}''',

    "multi_news" : '''{% set docs = document.split("3ed2dface8203c4c9dfb1a5dc58e41e0||") | reject("equalto",
      "") | list %}

      What are the key points across these news articles:

      {% for doc in docs %}


      Article: {{doc}}

      {% endfor %}

      |||

      {{summary[2:]}}''',

    # Generation
    "e2e_nlg_cleaned" : '''Combine all of the following data into a concise and grammatically correct
      text:

      {% for feature in meaning_representation.split("]") %} {% set key = feature.split("[")[0].replace(",","")
      %} {% set value = feature.replace(",","").replace(key+"[", '''''''') %}

      {% if value != "" %} {{key}} : {{value}} {% endif %}

      {%- endfor %}

      ||| {{human_reference}}''',

    "web_nlg_en" : '''Verbalize the following triples separated by a comma: {{old_input | join(\", \")}} ||| {{target}}''',

    "common_gen" : '''Ignoring the order of the concepts: {{ concepts | join(\", \") }}; \n\
      Generate a sentence with all the concepts :\n|||\n{{target}}''',

    # Sentiment Classification
    "imdb" : '''{{text}} Did the reviewer find this movie {{"good or bad"}}? ||| {{ answer_choices
      [label] }}''',

    "yelp_review_full" : '''{{ text }}

      ===

      Based on that, my rating is ||| {{ answer_choices[label] }}''',

    # open-domain QA
    "social_i_qa" : '''I heard that {{context}}


      And I was wondering {{question}}


      |||


      {{answer_choices[label | int - 1]}}''',

    "wiki_qa" : '''I am verifying the answers generated by an automatic system to the following
      question: {{question}}

      Suggested answer: {{answer}}

      Should I validate this answer?

      |||

      {{answer_choices[label]}}''',

    "fullwiki" : '''Answer the following question, \"{{question}}\", using the information\
      \ provided below.\n\n{% for sents in context.sentences %}\n  - {{sents | join(\"\
      \")}}\n{% endfor %}\n||| \n{{answer}}''',

    # Conference Resolution
    "winogrande_debiased" : "{{sentence}}\nReplace the _ in the above sentence with the correct option:\
       \n- {{option1}}\n- {{option2}}\n|||\n{% if answer == '1' %} {{option1}} {%\
       else %} {{ option2 }} {% endif %}", 

    "quoref" : '''The answer to the question: {{question}} is inside the article: {{context}},
      can you guess it ?


      |||

      {{answers.text | choice}}'''

}
    
task_is_generative_task = {
    # GLUE
    "cola": False,
    "mnli": False,
    "mrpc": False,
    "qnli": False,
    "qqp":  False,
    "sst2": False,
    "stsb": False,
    "wnli": False,
    # SuperGLUE excluding "rte": ("premise", "hypothesis"), since it is the same as in GLUE
    "boolq":    False,
    "cb":       False, 
    "wic":      False, 
    "copa":     False,
    "multirc":  False,
    "wsc.fixed":False,
    "rte":      False,
    "record":   False,
    # Tweet Eval
    'emoji':           False,
    'emotion':         False,
    'hate':            False,
    'irony':           False,
    'offensive':       False,
    'sentiment':       False,
    'stance_abortion': False,
    'stance_atheism':  False,
    'stance_climate':  False,
    'stance_feminist': False,
    'stance_hillary':  False,
    # NLI Tasks
    "anli_r1": False,
    "anli_r2": False,
    "anli_r3": False,
    # Summerization,
    "xsum": False,
    "samsum": False,
    "multi_news": False,
    # Generation
    "e2e_nlg_cleaned": True,
    "web_nlg_en": True,
    "common_gen": True,
    # Sentiment Classification
    "imdb": False,
    "yelp_review_full": False,
    # open-domain QA
    "social_i_qa": False,
    "wiki_qa": False,
    "fullwiki": False,
    # Coreference Resolution
    "winogrande_debiased": False,
    "quoref": False
}


# # Tweet Eval
#     'emoji':           "tweet_eval",
#     'emotion':         "tweet_eval",
#     'hate':            "tweet_eval",
#     'irony':           "tweet_eval",
#     'offensive':       "tweet_eval",
#     'sentiment':       "tweet_eval",
#     'stance_abortion': "tweet_eval",
#     'stance_atheism':  "tweet_eval",
#     'stance_climate':  "tweet_eval",
#     'stance_feminist': "tweet_eval",
#     'stance_hillary':  "tweet_eval",
#     # NLI Tasks
#     # "anli_r1": "anli",
#     # "anli_r2": "anli",
#     # "anli_r3": "anli",

# # Tweet Eval
#     "emoji" : '''Which emoji among {{answer_choices | join(", ")}} best describes the sentiment
#       of the following tweet?


#       {{text}} |||

#       {{answer_choices[label]}}''',

#     "emotion" : '''{{text}}


#       To get full credit in this exam, choose the correct emotion from the following
#       choices: {{answer_choices | join(", ")}}

#       |||

#       {{answer_choices[label]}}''',

#     "hate" : '''Does this tweet convey the author''s hatred towards something or someone?


#       {{text}} |||

#       {{answer_choices[label]}}''',

#     "irony" : '''Is this tweet is ironic? \n\n{{text}} |||\n{{answer_choices[label]}}''',

#     "offensive" : '''Can the tweet be removed for being offensive? Answer with a yes or a no.\
#       \ \n\n{{text}}\n|||\n{{answer_choices[label]}}''',

#     "sentiment" : '''Suppose you are the moderator of Twitter, what would be the sentiment\
#       \ of the following tweet: \n\n{{text}}\n\nOptions: {{answer_choices | join(\"\
#       , \")}}\n|||\n{{answer_choices[label]}}''',

#     "stance_abortion" : '''Does the author express any stance about abortion in the following text?


#       {{text}} |||

#       {{answer_choices[label]}}''',

#     "stance_atheism" : '''How would you describe the stance used in this tweet? {{answer_choices|join(", ")}}


#       {{text}} |||

#       {{answer_choices[label]}}''',

#     "stance_climate" : '''How would you describe the stance used in this tweet? {{answer_choices|join(", ")}}


#       {{text}} |||

#       {{answer_choices[label]}}''',

#     "stance_feminist" : '''How would you describe the stance used in this tweet? {{answer_choices|join(", ")}}


#       {{text}} |||

#       {{answer_choices[label]}}''',

#     "stance_hillary" : '''Does the author express any stance about Hillary in the following text?


#       {{text}} |||

#       {{answer_choices[label]}}''', 

#     # NLI Tasks
#     "anli_r1" : '''Suppose it's true that {{premise}} Then, is "{{hypothesis}}" {{"always"}},
#       {{"sometimes"}}, or {{"never"}} true? ||| {{ answer_choices[label] }}''',

#     "anli_r2" : '''Suppose it's true that {{premise}} Then, is "{{hypothesis}}" {{"always"}},
#       {{"sometimes"}}, or {{"never"}} true? ||| {{ answer_choices[label] }}''',
    
#     "anli_r3" : '''Suppose it's true that {{premise}} Then, is "{{hypothesis}}" {{"always"}},
#       {{"sometimes"}}, or {{"never"}} true? ||| {{ answer_choices[label] }}''',
