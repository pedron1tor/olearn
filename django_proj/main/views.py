from django.shortcuts import render, redirect
from django.contrib import messages
from sqlalchemy import inspect
import sqlalchemy
import pandas as pd
import ast
import numpy as np
from sqlalchemy.sql import exists
import xgboost as xgb
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as po
import plotly
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# function for processing the "lessons" dataframe
def lessons_process(lessons):
    # turn strdict into dict, assign to same column variable
    lessons['resources'] = [ast.literal_eval(x) for x in lessons['resources']]
    dataframes = []
    lesson_id_iter = iter(list(lessons['id']))
    for row in lessons['resources']:
        row_dict = {'content_id': [], 'channel_id': [], 'contentnode_id': [], }
        try:
            for diction in row:
                keys = diction.keys()
                for key in keys:
                    row_dict[key].append(diction[key])
            dataframe = pd.DataFrame(row_dict)
            dataframe['lesson_id'] = next(lesson_id_iter)
            dataframes.append(dataframe)
        except Exception as err:
            print(err)
            pass
    dataframe_1 = dataframes[0]
    for dataframe in dataframes[1:]:
        dataframe_1 = pd.concat([dataframe_1, dataframe], axis=0)
    final_merge = pd.merge(lessons, dataframe_1, left_on='id', right_on='lesson_id', how='inner')
    final_merge['difficulty'] = [x.split()[1] if x != '' else np.NaN for x in final_merge['description']]
    final_merge['subject'] = [x.split()[0] if x != '' else np.NaN for x in final_merge['description']]
    return final_merge


# Create your views here.
def menu(request):
    sql = """
        SELECT *
        FROM kolibriauth_facilityuser
        """
    db_conn = sqlalchemy.create_engine('sqlite:///path\\to\\kolibri\\db.sqlite3')
    local_db_conn = sqlalchemy.create_engine('sqlite:///db.sqlite3')
    from sqlalchemy import MetaData
    db_conn.connect()
    request.session['user'] = []
    facilusers = pd.read_sql(sql, db_conn)
    users = [str(x) for x in facilusers['username']]

    if local_db_conn.dialect.has_table(local_db_conn, "facilityuserstable"):
        pass

    else:
        facilusers['survey'] = 0
        facilusers.to_sql('facilityuserstable', local_db_conn, if_exists='replace')
    if request.method == 'POST':
        # if user == admin user
        if request.POST['users'] == 'pn1eto':
            request.session['user'] = request.POST['users']
            return redirect('admin_dashboard/')
        else:
            request.session['user'] = request.POST['users']
            print(facilusers)
            messages.success(request, f'Hola, ahora estás en tu cuenta {str(request.POST["users"])}')
            return redirect('dashboard/')

    return render(request, 'menu.html', {'users': users, })


def dashboard(request):
    localengine = sqlalchemy.create_engine('sqlite:///db.sqlite3')
    sql = """
            SELECT *
            FROM facilityuserstable
            """
    user_local = pd.read_sql(sql, localengine)
    if int(user_local[user_local['username'] == str(request.session['user'])]['survey']) == 0:
        return redirect('/survey')

    else:
        engine2 = sqlalchemy.create_engine('sqlite:///path\\to\\kolibri\\db.sqlite3')
        sql = """
                SELECT *
                FROM kolibriauth_facilityuser
                """
        facilusers = pd.read_sql(sql, engine2)
        sql = """
                SELECT *
                FROM logger_contentsessionlog
                """
        lessons_log = pd.read_sql(sql, engine2)
        sql = """
                SELECT *
                FROM logger_contentsummarylog
                """
        contentsummary = pd.read_sql(sql, engine2)
        sql = """
                SELECT *
                FROM content_contentnode
                where available = 1
                """
        channelcont = pd.read_sql(sql, engine2)
        sql = """
                SELECT *
                FROM logger_attemptlog
                """
        attempts = pd.read_sql(sql, engine2)
        sql = """
            SELECT *
            FROM lessons_lesson
            """
        lessons = pd.read_sql(sql, engine2)
        lessons = lessons_process(lessons)
        lessons_meta = pd.merge(lessons_log, facilusers, right_on='id', left_on='user_id', how='inner')
        lessons_meta = pd.merge(lessons_meta, channelcont, on='content_id', how='inner', )
        lessons_meta = pd.merge(lessons_meta, contentsummary, on='content_id', how='inner', )
        lessons_meta = pd.merge(lessons_meta, lessons, on='content_id', )
        lessons_meta['video_loc'] = np.NaN
        video_loc = [0 if x == '{}' else ast.literal_eval(x)['contentState']['savedLocation'] for x in
                     lessons_meta[lessons_meta['kind'] == 'video']['extra_fields_y']]
        lessons_meta.loc[lessons_meta['kind'] == 'video', 'extra_fields_y'] = video_loc
        materias = set([x for x in lessons_meta['subject'].dropna(axis=0)])
        lessons_detailed = lessons.groupby('title').sum()
        lessons_detailed = lessons_detailed.rename({'is_active': 'number_resources', }, axis='columns')
        lessons_detailed = lessons_detailed.drop(['_morango_dirty_bit'], axis=1)
        lessons_detailed = pd.merge(lessons_detailed,
                                    lessons[['id', 'difficulty', 'subject', 'title']].drop_duplicates(subset='id'),
                                    on='title'
                                    , how='left')

        # todo add user sorting
        # todo remove changed or deleted lessons
        lessons_meta = lessons_meta[lessons_meta.title_y != 'Segundo grado - Decenas y centenas']

        lessons_meta_agg = lessons_meta.drop_duplicates(subset='id_y').groupby('title_y').sum()
        lessons_detailed = pd.merge(lessons_detailed, lessons_meta_agg, left_on='title', right_on='title_y', how='left')
        lessons_detailed['Completed'] = lessons_detailed['number_resources'] - lessons_detailed['progress_x']
        lessons_detailed['Completed'] = [1 if x == 0 else 0 for x in lessons_detailed['Completed']]
        # todo add video watch and right exercises to the mix
        lessons_detailed['video_watch'] = lessons_detailed['time_spent_x'] / lessons_detailed['video_loc']
        lessons_detailed['video_watch'] = (lessons_detailed['video_watch'] * 100) / 1.5
        localengine = sqlalchemy.create_engine('sqlite:///db.sqlite3')
        lessons_detailed.to_sql('curr_iter_detailed_lessons', localengine, if_exists='replace')
        lessons_detailed.to_csv('datasets/lessons_detailed_merged.csv')
        lessons_completed = lessons_detailed[lessons_detailed.Completed == 1]
        lessons_completed = [x for x in lessons_completed['title']]
        # lessons_detailed = pd.merge(lessons_detailed, l)
        # lessons_detailed.to_csv('lessons_agg.csv')
        # todo fix sorting by subject
        # plots
        import plotly.express as px
        import plotly.io as pio
        import plotly.graph_objs as po
        import plotly
        fig = plotly.graph_objs.Figure()

        # Add traces
        fill_na_df = lessons_detailed.fillna(value=0)
        fig.add_trace(
            po.Scatter(x=[str(x) for x in lessons_detailed.title], y=[x for x in fill_na_df.progress_x],
                       name='Materials completed'))
        fig.add_trace(po.Scatter(x=[str(x) for x in lessons_detailed.title], y=[x for x in fill_na_df.number_resources],

                                 name='Number of materials', line=dict(color='Red')))
        fig.update_layout(title='Progress on lessons', showlegend=True)
        plot = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
        try:

            lesson_difficulty = int(list(lessons_detailed['difficulty'].dropna(axis='rows'))[-1]) + int(
                request.session['lesson_level'])
            recommended_lesson = lessons_detailed.loc[lessons_detailed['difficulty'] == str(lesson_difficulty), :]
            return render(request, 'dashboard.html',
                          {'user': str(request.session['user']), 'materias': materias, 'lessons': lessons_completed,
                           'plot': plot,
                           'recommended_lesson': list(recommended_lesson.title)[0]})
        except Exception as err:
            print(err)
            pass
    if request.method == 'POST':
        request.session['lesson'] = request.POST['lesson']
        return redirect('/lesson_eval')
    return render(request, 'dashboard.html',
                  {'user': str(request.session['user']), 'materias': materias, 'lessons': lessons_completed,
                   'plot': plot,
                   })


def survey(request):
    if request.method == 'POST':
        localengine = sqlalchemy.create_engine('sqlite:///db.sqlite3')
        messages.success(request, 'Tus respuestas fueron guardadas')
        sql = """
                    SELECT *
                    FROM facilityuserstable
                    """
        db_df = pd.read_sql(sql, localengine)
        info_dict = {'username': request.session['user'], 'school': request.POST['school'],
                     'age': request.POST['age'], 'paternal_status': request.POST['paternal_status'],
                     'study_time': request.POST['study_time'],
                     'grade_average': request.POST['grade_average']
            , 'free_time': request.POST['free_time'],
                     'job_mother': request.POST['job_mother']
            , 'job_father': request.POST['job_father']
            , 'extra_activities': request.POST['extra_activities'],
                     'alcohol': request.POST['alcohol'],
                     'drugs': request.POST['drugs']}
        info_df = pd.DataFrame(info_dict, index=[0])
        db_df.loc[db_df['username'] == str(request.session['user']), ['survey']] = 1
        try:
            print(db_df['school'])
            for col in list(info_df.columns):
                print(col)
                user = request.session['user']
                db_df.index = db_df.username
                db_df.loc[user, col] = list(info_df[col])[0]

            db_df = db_df.reset_index(drop=True)
            print(db_df)
            print(db_df.columns)
            db_df = db_df.drop('level_0', axis='columns')
            db_df.to_sql('facilityuserstable', localengine, if_exists='replace')
        except Exception as err:
            print(err)
            merged_df = pd.merge(db_df, info_df, left_on='username', right_on='username', how='left')
            print(merged_df)
            merged_df.to_sql('facilityuserstable', localengine, if_exists='replace')

        # merged_df.to_csv('merge_from_survey.csv')
        # todo fix innecesary merging
        return redirect('/dashboard')
    return render(request, 'survey.html', {})


def evaluation(request):
    if request.method == 'POST':

        localengine = sqlalchemy.create_engine('sqlite:///db.sqlite3')

        messages.success(request, 'Tus respuestas fueron guardadas')
        if localengine.dialect.has_table(localengine, "lesson_feedback"):
            # todo fix for new lessons in curr_iter_detailed_lessons
            sql = """
                                            SELECT *
                                            FROM curr_iter_detailed_lessons
                                            """
            db_df = pd.read_sql(sql, localengine)
            sql = """
                                                        SELECT *
                                                        FROM lesson_feedback
                                                        """
            info_df_db = pd.read_sql(sql, localengine)
            info_dict = {'lesson': request.session['lesson'], 'difficulty': request.POST['difficulty'],
                         'liking': request.POST['like'], 'easiness_vid': request.POST['easiness_vid'],
                         'easiness_ex': request.POST['easiness_ex'],
                         'study_time': request.POST['study_time'], }
            info_df = pd.DataFrame(info_dict, index=[0])
            info_df['difficulty'] = [int(x) for x in info_df['difficulty']]
            info_df['easiness_ex'] = [int(x) for x in info_df['easiness_ex']]
            result_df = pd.merge(info_df_db, info_df, right_on='lesson', left_on='title', how='left')
            score_sum = sum(list(info_df.loc[0, ['difficulty', 'easiness_ex']]))
            if score_sum == 4:
                info_df['recommended'] = -2
                request.session['lesson_level'] = -2
            elif 2 <= score_sum >= 3:
                info_df['recommended'] = -1
                request.session['lesson_level'] = -1
            elif score_sum <= 1:
                info_df['reccomended'] = 1
                request.session['lesson_level'] = 1
            result_df = result_df.drop('level_0', axis='columns')
            result_df.to_sql('lesson_feedback', localengine, if_exists='replace')
            return redirect('/dashboard')

        else:
            sql = """
                                SELECT *
                                FROM curr_iter_detailed_lessons
                                """
            db_df = pd.read_sql(sql, localengine)
            info_dict = {'lesson': request.session['lesson'], 'difficulty': request.POST['difficulty'],
                         'liking': request.POST['like'], 'easiness_vid': request.POST['easiness_vid'],
                         'easiness_ex': request.POST['easiness_ex'],
                         'study_time': request.POST['study_time'], }
            info_df = pd.DataFrame(info_dict, index=[0])
            info_df['difficulty'] = [int(x) for x in info_df['difficulty']]
            info_df['easiness_ex'] = [int(x) for x in info_df['easiness_ex']]
            score_sum = sum(list(info_df.loc[0, ['difficulty', 'easiness_ex']]))
            if score_sum == 4:
                info_df['recommended'] = -2
                request.session['lesson_level'] = -2
            elif 2 <= score_sum >= 3:
                info_df['recommended'] = -1
                request.session['lesson_level'] = -1
            elif score_sum <= 1:
                info_df['reccomended'] = 1
                request.session['lesson_level'] = 1
            result_df = pd.merge(db_df, info_df, right_on='lesson', left_on='title', how='left')
            result_df.to_sql('lesson_feedback', localengine, if_exists='replace')
            return redirect('/dashboard')
    return render(request, 'evaluation_lesson.html', {'lesson': request.session['lesson']})


def query_table(conn, table):
    sql = f'select * from {table}'
    df = pd.read_sql(sql, conn)
    return df


from .heatmap import GetCorr


def admin_dash(request):
    # data generation for demonstration purposes
    from sklearn.datasets import make_classification
    all_data = pd.read_csv(r'path_to_sample_data',
                           index_col=0)
    print(all_data.columns)
    X, y = all_data.loc[:, all_data.columns != 'at_risk_target'], all_data['at_risk_target']
    x_data = pd.DataFrame(data=X)
    print(x_data)
    model = xgb.XGBRegressor()
    model.fit(X, y)
    importance = model.feature_importances_
    model2 = LogisticRegression()
    model2.fit(X, y)
    importance = model2.coef_[0]
    fig = plotly.graph_objs.Figure()

    fig.add_trace(
        po.Bar(x=X.columns, y=importance,
               name='Feature importance'))
    # fig.add_trace(po.Scatter(x=[str(x) for x in lessons_detailed.title], y=[x for x in fill_na_df.number_resources],

    #                name='Number of materials', line=dict(color='Red')))
    fig.update_layout(title='Feature importance', showlegend=True)
    feat_imp = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
    freq_table = pd.crosstab(index=all_data.at_risk_target, columns=all_data.grade_average)
    print(freq_table)
    # clustering
    from sklearn.datasets.samples_generator import make_blobs
    # all_data.to_csv(r'make_class_dataset.csv')
    # all_data['scaled_average_grade'] = np.array(((100-0)*(all_data['average_grade']-all_data['average_grade'].min()))/(all_data['average_grade'].max())-(all_data['average_grade'].min())).clip(0, 100)
    if request.method == 'POST':
        correlation = GetCorr(all_data, 'Student data', int(len(all_data.corr().columns))).GetHeatmap()
        return redirect('/heatmap')
    return render(request, 'admin_dash.html', {'plot': feat_imp})


def heatmap(request):
    return render(request, 'myPlot.html', {})
