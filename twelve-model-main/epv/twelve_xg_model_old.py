"""Transcribing the expected goals model implemented in the Twelve website."""
import math
import numpy as np

from opta.settings import ROOT_DIR


def load_EPV_grid(fname='EPV_grid.csv'):
    """ load_EPV_grid(fname='EPV_grid.csv')

    # load pregenerated EPV surface from file.

    Parameters
    -----------
        fname: filename & path of EPV grid (default is 'EPV_grid.csv' in the curernt directory)

    Returns
    -----------
        EPV: The EPV surface (default is a (32,50) grid)

    """
    epv = np.loadtxt(ROOT_DIR +'/models/'+ fname, delimiter=',')
    return epv


EPV = load_EPV_grid()
def get_EPV_at_location(position, field_dimen=(100, 100)):

    x, y = position

    ny, nx = EPV.shape
    xmin = 0
    ymin = 0

    xi = (x - xmin) / field_dimen[0] * nx
    yj = (y - ymin) / field_dimen[1] * ny
    xi = max([round(xi)-1,0])
    yj = max([round(yj)-1,0])
    return EPV[int(yj), int(xi)]


def expected_goals_modified(x_position, y_position, event_qualifiers) -> float:

    if y_position > 50:
        y_position = 100 - y_position
    adjusted_x = 1.05 * x_position
    adjusted_y = 0.68 * y_position
    distance = math.sqrt(math.pow(105 - adjusted_x, 2) + math.pow(34 - adjusted_y, 2))
    var_a = math.sqrt(math.pow(105 - adjusted_x, 2) + math.pow(37.66 - adjusted_y, 2))
    var_b = math.sqrt(math.pow(105 - adjusted_x, 2) + math.pow(30.34 - adjusted_y, 2))
    var_z = (math.pow(var_a, 2) + math.pow(var_b, 2) - math.pow(7.32, 2)) / (2 * var_a * var_b)
    angle = math.acos(var_z) * 180 / math.pi

    strong = any(int(q) == 113 for q in event_qualifiers)
    weak = any(int(q) == 114 for q in event_qualifiers)
    big_chance = any(int(q) == 214 for q in event_qualifiers)
    head = any(int(q) == 15 for q in event_qualifiers)
    swerve_left = any(int(q) == 120 for q in event_qualifiers)
    swerve_right = any(int(q) == 121 for q in event_qualifiers)
    swerve_moving = any(int(q) == 122 for q in event_qualifiers)
    individual_play = any(int(q) == 215 for q in event_qualifiers)
    fast_break = any(int(q) == 23 for q in event_qualifiers)
    volley = any(int(q) == 108 for q in event_qualifiers)

    if any(int(q) == 26 for q in event_qualifiers):
        return expected_direct_free_kick(distance, swerve_right, swerve_left, strong)
    if any(int(q) == 24 for q in event_qualifiers):
        return expected_indirect_free_kick(distance, big_chance, head, strong, angle)
    if any(int(q) == 25 for q in event_qualifiers):
        return expected_corner(distance, big_chance, head, strong, angle, volley)
    if any(int(q) == 9 for q in event_qualifiers):
        return expected_penalty()
    return expected_shot(distance, big_chance, head, strong, weak, angle, swerve_right, swerve_left,
                         swerve_moving, individual_play, fast_break)


def expected_goals(shot_event) -> float:
    """
    Returns the xG value for a shot event. The value is determined
    based on the qualifiers and position of the shot.
    Notes:
        This is the main function for getting xg.
    """
    x_position = float(shot_event.get('x'))
    y_position = float(shot_event.get('y'))
    if y_position > 50:
        y_position = 100 - float(shot_event.get('y'))
    adjusted_x = 1.05 * x_position
    adjusted_y = 0.68 * y_position
    distance = math.sqrt(math.pow(105 - adjusted_x, 2) + math.pow(34 - adjusted_y, 2))
    var_a = math.sqrt(math.pow(105 - adjusted_x, 2) + math.pow(37.66 - adjusted_y, 2))
    var_b = math.sqrt(math.pow(105 - adjusted_x, 2) + math.pow(30.34 - adjusted_y, 2))
    var_z = (math.pow(var_a, 2) + math.pow(var_b, 2) - math.pow(7.32, 2)) / (2 * var_a * var_b)
    angle = math.acos(var_z) * 180 / math.pi

    event_qualifiers = shot_event.findall('Q')

    strong = any(int(q.get('qualifier_id')) == 113 for q in event_qualifiers)
    weak = any(int(q.get('qualifier_id')) == 114 for q in event_qualifiers)
    big_chance = any(int(q.get('qualifier_id')) == 214 for q in event_qualifiers)
    head = any(int(q.get('qualifier_id')) == 15 for q in event_qualifiers)
    swerve_left = any(int(q.get('qualifier_id')) == 120 for q in event_qualifiers)
    swerve_right = any(int(q.get('qualifier_id')) == 121 for q in event_qualifiers)
    swerve_moving = any(int(q.get('qualifier_id')) == 122 for q in event_qualifiers)
    individual_play = any(int(q.get('qualifier_id')) == 215 for q in event_qualifiers)
    fast_break = any(int(q.get('qualifier_id')) == 23 for q in event_qualifiers)
    volley = any(int(q.get('qualifier_id')) == 108 for q in event_qualifiers)

    if any(int(q.get('qualifier_id')) == 26 for q in event_qualifiers):
        return expected_direct_free_kick(distance, swerve_right, swerve_left, strong)
    if any(int(q.get('qualifier_id')) == 24 for q in event_qualifiers):
        return expected_indirect_free_kick(distance, big_chance, head, strong, angle)
    if any(int(q.get('qualifier_id')) == 25 for q in event_qualifiers):
        return expected_corner(distance, big_chance, head, strong, angle, volley)
    if any(int(q.get('qualifier_id')) == 9 for q in event_qualifiers):
        return expected_penalty()
    return expected_shot(distance, big_chance, head, strong, weak, angle, swerve_right, swerve_left,
                         swerve_moving, individual_play, fast_break)


def expected_direct_free_kick(distance: float, swerve_right: bool,
                              swerve_left: bool, strong: bool) -> float:
    """expected goals direct free kicks"""
    coefficient = (-0.2106 * distance + 0.0026 * math.pow(distance, 2) +
                   1.0941 * swerve_right + 1.4534 * swerve_left + 1.6874 * strong)
    return math.exp(coefficient) / (1 + math.exp(coefficient))


def expected_indirect_free_kick(distance: float, big_chance: bool,
                                head: bool, strong: bool, angle: float) -> float:
    """expected goals indirect free kicks"""
    coefficient = (-2.1170 - 0.0787 * distance + 2.1786 *
                   big_chance - 0.5325 * head + 2.3404 * strong + 0.0144 * angle)
    return math.exp(coefficient) / (1 + math.exp(coefficient))


def expected_corner(distance: float, big_chance: bool, head: bool,
                    strong: bool, angle: float, volley: bool) -> float:
    """expected goals corners"""
    coefficient = (-2.5621 - 0.0671 * distance + 0.0229 * angle + 1.4080 *
                   big_chance - 0.5836 * head + 1.8419 * strong - 0.2779 * volley)
    return math.exp(coefficient) / (1 + math.exp(coefficient))


def expected_penalty() -> float:
    """expected goals penalty"""
    return 0.751865671642


def expected_shot(distance: float, big_chance: bool, head: bool, strong: bool,
                  weak: bool, angle: float, swerve_right: bool, swerve_left: bool,
                  swerve_moving: bool, individual_play: bool, fast_break: bool) -> float:
    """expected goals shots"""
    swerve = False
    if swerve_right or swerve_left or swerve_moving:
        swerve = True
    coefficient = (-3.2199 + 1.8054 * big_chance - 0.5363 * head + 0.8151 * fast_break
                   + 0.3287 * individual_play + 1.8024 * strong - 1.4474 * weak
                   - 1.5220 * strong * big_chance + 0.7355 * swerve + 0.0187 * angle
                   - 0.0032 * math.pow(distance, 2) + 0.0021 * distance * angle)
    return math.exp(coefficient) / (1 + math.exp(coefficient))


def xT_pass(x1, y1, x2, y2, cross=False, throughBall=False, pullBack=False, chanceCreated=False, flickOn=False):

    adjustedY2 = y2 if y1 <= 50 else 100 - y2
    adjustedY1 = y1 if y1 <= 50 else 100 - y1

    coefficient = -3.720748 + 6.037517e-03 * x1 + 3.584304e-02 * x2 + 2.617176e-02 * adjustedY2 - 6.156030e-04 * x1 * adjustedY1 + 5.304036e-04 * math.pow(
        adjustedY1, 2) - 3.027166e-04 * math.pow(x1, 2) - 7.325353e-06 * math.pow(adjustedY1,
                                                                                  3) + 4.716508e-06 * math.pow(
        x1, 3) - 4.951233e-04 * x2 * adjustedY2 - 4.466221e-04 * math.pow(x2, 2) + 1.128160e-04 * math.pow(
        adjustedY2,
        2) - 4.959944e-06 * math.pow(
        adjustedY2, 3) + 4.849165e-06 * math.pow(x2,
                                                 3) + 2.196781e-04 * x1 * x2 - 4.024221e-04 * adjustedY1 * adjustedY2 + 1.057939e-05 * x1 * adjustedY1 * x2 + 4.241092e-06 * x1 * x2 * adjustedY2 - 8.232459e-08 * x1 * adjustedY1 * x2 * adjustedY2 - 6.654025e-06 * math.pow(
        x1, 2) * x2 + 4.668466e-06 * math.pow(x2, 2) * adjustedY2 + 7.636041e-06 * math.pow(adjustedY2,
                                                                                            2) * adjustedY1
    logistic = (math.exp(coefficient) / (1 + math.exp(coefficient)))
    linear = 7.628156e-03 * x1 + 7.996155e-03 * x2 + 1.445358e-03 * adjustedY2 - 1.368979 * cross + 8.410532e-01 * throughBall + 7.921517e-02 * pullBack - 9.274986e-02 * chanceCreated - 7.581955e-05 * x1 * adjustedY1 - 4.716755e-05 * math.pow(
        x1, 2) - 9.534056e-05 * x2 * adjustedY2 - 6.851437e-05 * math.pow(x2, 2) + 7.821691e-07 * math.pow(x2,
                                                                                                           3) - 2.111737e-04 * x1 * x2 + 5.654900e-05 * adjustedY1 * adjustedY2 + 5.308242e-07 * x1 * adjustedY1 * x2 + 2.328050e-07 * x1 * x2 * adjustedY2 + 1.423698e-02 * cross * x2 - 5.765683e-03 * throughBall * x2 + 3.073823e-03 * flickOn * x2 - 3.470719e-03 * flickOn * x1 + 1.094886e-01 * chanceCreated * cross + 7.758500e-02 * chanceCreated * flickOn - 9.178206e-02 * chanceCreated * throughBall + 4.158375e-07 * math.pow(
        x1, 2) * x2 + 7.818592e-07 * math.pow(x2, 2) * x1 + 3.818770e-07 * math.pow(x1,
                                                                                    2) * adjustedY1 + 8.122093e-07 * math.pow(
        x2, 2) * adjustedY2 - 4.910344e-07 * math.pow(adjustedY2, 2) * adjustedY1

    res = logistic * linear
    if res > 1:
        return 1
    return max(0, res)