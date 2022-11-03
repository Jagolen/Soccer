import pandas as pd


def create_base_dataset(start_x=50, start_y=50, one_dim=True, simple=False):
    data = list()
    for _x2 in range(0, 100):
        for _y2 in range(0, 100):
            row = {
                'start_x': start_x,
                'start_y': start_y,
                'end_x': _x2,
                'end_y': _y2,
            }
            data.append(row)

    df = pd.DataFrame(data)

    if simple:
        return df

    if one_dim:
        df[['start_x', 'start_y']] = df[['end_x', 'end_y']]

    df = feature_creation(df)

    df['const'] = 1

    return df


def distance_to_goal(x, y):
    newx = float(x * 1.05)
    newy = float(y * 0.68)
    return ((34 -newy)**2 + (105 - newx)**2)**0.5


def pass_distance(x1, y1, x2, y2):
    newx = float(x1 * 1.05)
    newy = float(y1 * 0.68)
    return ((y2 * 0.68 - newy) ** 2 + (x2 * 1.05 - newx) ** 2) ** 0.5


def feature_creation(df):

    if 'pass_length' not in df:
        df['pass_length'] =  [pass_distance(x1, y1, x2, y2) for x1, y1, x2, y2 in  zip(df['start_x'], df['start_y'], df['end_x'], df['end_y'])]

    if 'start_y_adj' not in df:
        df['start_y_adj'] = [100 - x if x > 50 else x for x in df['start_y']]

    if 'end_y_adj' not in df:
        df['end_y_adj'] = [100 - y2 if y1 > 50 else y2 for y1, y2 in df[['start_y', 'end_y']].values]

    df['distance_start'] = [distance_to_goal(x1, y1) for x1, y1 in  zip(df['start_x'], df['start_y'])]
    df['distance_end'] = [distance_to_goal(x1, y1) for x1, y1 in  zip(df['end_x'], df['end_y'])]
    df['directness'] = df['distance_start'] - df['distance_end']

    return df


if __name__ == '__main__':


    #df['pass_length_diff'] = df['pass_length_test'] - df['pass_length']
    #df = df[df['pass_length_diff'] < 5]

    # Add new Atts
    print()