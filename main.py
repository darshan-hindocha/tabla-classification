import librosa
import numpy as np
import os
import csv
import glob
import tqdm
import re


def compare_against_type(file_name, track_name):
    # Define a regular expression to match non-alphanumeric characters
    pattern = re.compile(r'\W+')

    # Use the regular expression to remove any non-alphanumeric characters
    result = re.sub(pattern, '', file_name).lower()

    return track_name in result


# Define the path to the teacher tracks
teacher_dir = "data/teacher/"

# Define the path to the student tracks
student_dir = "data/student/"

track_types = [
    "hinch",
    "dadr",
    "kervo1",
    "kervo2",
    "khemto",
    "pchandi",
]


# Define the function to compare two audio tracks
def compare_audio(original: np.ndarray, student: np.ndarray) -> float:
    original, _ = librosa.load(original)
    student, _ = librosa.load(student)

    # Set the tempo (beats per minute)
    tempo, _ = librosa.beat.beat_track(y=original)

    # Compute the beat frames for both tracks
    original_beats = librosa.onset.onset_detect(y=original, sr=44100, hop_length=512)
    student_beats = librosa.onset.onset_detect(y=student, sr=44100, hop_length=512)

    # Compute the difference between the beat frames
    beat_diff = np.abs(len(original_beats) - len(student_beats))

    # Compute the score based on the difference in the number of beats
    score = 1 - (beat_diff / len(original_beats))

    return score


# Get the list of all the teacher tracks
teacher_tracks = os.listdir(teacher_dir)

# Get the list of all the student folders
student_folders = [folder for folder in os.listdir(student_dir)]

# Loop through each student folder
for student_folder in tqdm.tqdm(student_folders):
    # Get the list of all the student tracks in the folder
    student_tracks = glob.glob(os.path.join(student_dir, student_folder, "*.m4a" or "*.wav" or "*.mp3" or "*.flac"))

    # Create a dictionary to store the scores for each student track
    scores_dict = {}

    for track_type in track_types:
        # get student track matching track type
        student_track = [track for track in student_tracks if compare_against_type(track, track_type)]
        if len(student_track) == 0:
            print(f"Student track not found for {track_type} in {student_folder}")
            continue
        student_track = student_track[0].split("/")[-1]

        # get teacher track matching track type
        teacher_track = [track for track in teacher_tracks if compare_against_type(track, track_type)][0]

        # Compute the score and store it in the dictionary
        score = compare_audio(
            os.path.join(teacher_dir, teacher_track),
            os.path.join(student_dir, student_folder, student_track)
        )

        scores_dict[(teacher_track, student_track)] = score

    # Save the scores as a CSV file for each student
    output_path = f"data/scores/{student_folder}.csv"
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Teacher Track", "Student Track", "Score"])
        for (teacher_track, student_track), score in scores_dict.items():
            writer.writerow([teacher_track, student_track, score])

if __name__ == "__main__":
    print("Done")
