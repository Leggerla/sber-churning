def months_processing(df):
    import pandas as pd

    df_copy = df.copy()

    df_copy.body.fillna('empty', inplace=True)

    df_copy_array = df_copy.values

    find_empty_mess = []
    for i in range(len(df_copy_array)):
        subject_str = df_copy_array[i][2]
        body_str = df_copy_array[i][3]

        if subject_str == '0' and body_str == 'empty':
            find_empty_mess.append(1)
        else:
            find_empty_mess.append(0)

    df_copy['empty_msgs'] = find_empty_mess

    dummy = pd.get_dummies(df_copy['type'])
    df_copy['email'] = dummy['email']
    df_copy['push'] = dummy['push']
    df_copy['sms'] = dummy['sms']
    df_copy = df_copy.drop(['subject', 'body', 'type', 'day',
                            'wday', 'hours', 'minutes', 'count'], axis=1)

    main_dataframe = pd.DataFrame()

    print(df_copy['year'].unique())

    for time_season in range(1, 12):
        df_season = df_copy[df_copy['month'] == time_season]
        df_season = df_season.drop(['action_id', 'year', 'month'], axis=1)
        df_season = df_season.groupby(['user_id'], as_index=False).sum()
        df_season['month'] = time_season
        main_dataframe = main_dataframe.append(df_season)

    return main_dataframe