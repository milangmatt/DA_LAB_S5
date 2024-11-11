Familiarization of Python
==

Aim
--
To familiarize with the basic concepts of python programming language

Theory
--
Python is a high-level, interpreted programming language that is widely used for various purposes. Here are some of its use cases and fields of usage:

### Use Cases:

* **Web Development**: Python is used in web development to build web applications and web services using popular frameworks like Django and Flask.
* **Data Analysis and Science**: Python is widely used in data analysis, machine learning, and data science due to its simplicity and the availability of libraries like NumPy, pandas, and scikit-learn.
* **Automation**: Python is used to automate tasks, such as data entry, file management, and system administration, due to its easy-to-learn syntax and vast number of libraries.
* **Scientific Computing**: Python is used in scientific computing for tasks like data analysis, numerical simulations, and data visualization due to its simplicity and the availability of libraries like NumPy and SciPy.
* **Education**: Python is often taught in introductory programming courses due to its simplicity and ease of use.

### Fields of Usage:

* Artificial Intelligence and Machine Learning
* Data Science and Analytics
* Web Development
* Automation and Scripting
* Scientific Computing and Research
* Education and Academia
* Network Security and Penetration Testing
* Game Development
* Desktop Applications and GUI Programming


----
### Data Types in Python

### 1. Numeric Types
*   Integers
*   Floating Point Numbers
*   Complex Numbers

### 2. Sequence Types
*   Strings
*   Lists
*   Tuples

### 3. Mapping Type
*   Dictionaries

### 4. Boolean Type
*   Boolean

### 5. Set Types
*   Sets
*   Frozensets

### 6. None Type
*   None

### 7. Binary Types
*   Bytes
*   ByteArray
*   MemoryView

-----

### Arithmetic Operations in Python

*   Addition: `a + b`
*   Subtraction: `a - b`
*   Multiplication: `a * b`
*   Division: `a / b`
*   Floor Division: `a // b`
*   Modulus: `a % b`
*   Exponentiation: `a ** b`

----
### Python Conditional Statement
```python
if condition:
    # code to execute if condition is true
elif another_condition:
    # code to execute if another_condition is true
else:
    # code to execute if all conditions are false
```
---

### Python Loops

### For Loop

The for loop in Python is used to iterate over a sequence (such as a list, tuple, dictionary, set, or string) or other iterable objects.

1. List Iterator
```python
for index, value in iterable:
    # code to execute
```

  2. Range Iterator
```python
for value in range(start, end, step):
    # code to execute
```
### While Loop

The while loop in Python is used to execute a block of code as long as a certain condition is true.

```python
# Syntax
while condition:
    # code to execute
```

---

### Functions in Python

Functions in Python are blocks of code that can be executed multiple times from different parts of a program. They are useful for organizing code, reducing repetition, and making code more reusable.

### Basic Syntax

The basic syntax of a function in Python is as follows:

```python
def function_name(parameters):
    # code to execute
```

*   `def` is the keyword used to define a function.
*   `function_name` is the name of the function.
*   `parameters` are the values passed to the function when it is called.
*   The code to execute is the block of code inside the function.
---

### Basic Data Types in Python

#### 1. List
A list is a collection of items that can be of any data type, including strings, integers, floats, and other lists.
*   Notation: `[]`
*   Example: `my_list = [1, 2, 3, "hello", 4.5]`

#### 2. Tuple
A tuple is a collection of items that can be of any data type, including strings, integers, floats, and other tuples. Tuples are immutable, meaning they cannot be changed after they are created.
*   Notation: `()`
*   Example: `my_tuple = (1, 2, 3, "hello", 4.5)`

#### 3. Set
A set is a collection of unique items that can be of any data type, including strings, integers, floats, and other sets.
*   Notation: `{}` or `set()`
*   Example: `my_set = {1, 2, 3, "hello", 4.5}`

#### 4. Dictionary
A dictionary is a collection of key-value pairs where each key is unique and maps to a specific value.
*   Notation: `{key: value}`
*   Example: `my_dict = {"name": "John", "age": 30, "city": "New York"}`
  

--
### Comments in Python

Comments in Python are used to explain the code and make it more readable. They are ignored by the Python interpreter.

### Types of Comments

There are two types of comments in Python:

#### 1. Single-Line Comments

Single-line comments start with the `#` symbol and continue until the end of the line.

```python
# This is a single-line comment
```

#### 2. Multi-Line Comments

Multi-line comments are enclosed in triple quotes `"""` or `'''`. They can span multiple lines.

```python
"""
This is a multi-line comment
that spans multiple lines
"""
```

Result
--
Python Programming Language and its basic concepts were familiarized