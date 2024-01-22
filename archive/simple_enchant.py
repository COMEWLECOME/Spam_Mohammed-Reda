import enchant
broker = enchant.Broker()
broker.describe()
broker.list_languages()


# Create an instance of the dictionary
dictionary = enchant.Dict("en_US")

# Check the spelling of a word
word = "hello"
if dictionary.check(word):
    print(f"{word} is spelled correctly")
else:
    print(f"{word} is spelled incorrectly")