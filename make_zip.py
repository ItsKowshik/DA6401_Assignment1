import shutil
import os

print("Checking for src folder...")
if not os.path.exists('src/train.py'):
    print("ERROR: I can't find 'src/train.py'. Make sure this script is NEXT TO the 'src' folder, not inside it!")
else:
    print("Zipping the src folder perfectly for Gradescope...")
    # root_dir='.' means start here
    # base_dir='src' means specifically wrap everything inside a folder named 'src'
    shutil.make_archive('submission', 'zip', root_dir='.', base_dir='src')
    print("SUCCESS! A file named 'submission.zip' was just created.")