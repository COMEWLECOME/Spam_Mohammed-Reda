# from spellchecker import SpellChecker #conda install conda-forge::pyspellchecker

# # Create an instance of the spell checker
# spell = SpellChecker()

# # Check the spelling of a word
# word = "helo"
# corrected_word = spell.correction(word)

# # Print the corrected word
# print(f"The corrected word for '{word}' is '{corrected_word}'.")


# import pandas as pd

def error_count(df):
    from spellchecker import SpellChecker
    spell = SpellChecker()
    error_count = []
    
    for text in df['mail']:
        words = text.split()
        count = 0
        for word in words:
            if not spell.correction(word) == word:
                count += 1
        error_count.append(count)

    df['nombre derreur'] = error_count
    return df

   
