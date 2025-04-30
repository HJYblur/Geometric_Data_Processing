import os
import pandas
from natsort import natsorted


def collate_grades(individual_grade_sheets: list[pandas.DataFrame]) -> pandas.DataFrame:
    # Combine all submission grades into a single table
    df = pandas.concat(individual_grade_sheets)
    scores = df['score'].apply(pandas.to_numeric, errors='coerce')
    times = df['time'].apply(pandas.to_numeric, errors='coerce')

    # The scores & times for each task are given by the mean of the associated tests
    scores = scores.unstack('test').mean(axis=1)
    times = times.unstack('test').mean(axis=1)

    # Re-combine score & time as columns
    df = pandas.DataFrame({
        'score': pandas.Series(scores, name='score'),
        'time': pandas.Series(times, name='time')
    })

    # Convert submissions IDs into columns
    df = df.unstack(level='submission')
    df = df.swaplevel(axis=1)
    df = df.reindex(natsorted(df.columns), axis=1)
    return df
