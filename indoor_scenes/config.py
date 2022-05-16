import argparse

ROOM_TYPES = [
    'airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar', 'bathroom',
    'bedroom', 'bookstore', 'bowling', 'buffet', 'casino', 'children_room',
    'church_inside', 'classroom', 'cloister', 'closet', 'clothingstore',
    'computerroom', 'concert_hall', 'corridor', 'deli', 'dentaloffice',
    'dining_room', 'elevator', 'fastfood_restaurant', 'florist', 'gameroom',
    'garage', 'greenhouse', 'grocerystore', 'gym', 'hairsalon', 'hospitalroom',
    'inside_bus', 'inside_subway', 'jewelleryshop', 'kindergarden', 'kitchen',
    'laboratorywet', 'laundromat', 'library', 'livingroom', 'lobby',
    'locker_room', 'mall', 'meeting_room', 'movietheater', 'museum', 'nursery',
    'office', 'operating_room', 'pantry', 'poolinside', 'prisoncell',
    'restaurant', 'restaurant_kitchen', 'shoeshop', 'stairscase',
    'studiomusic', 'subway', 'toystore', 'trainstation', 'tv_studio',
    'videostore', 'waitingroom', 'warehouse', 'winecellar'
]

USED_ROOM_TYPES = [
    'bathroom', 'bedroom', 'bookstore', 'children_room', 'closet', 'corridor',
    'dining_room', 'garage', 'gym', 'kitchen', 'laundromat', 'livingroom',
    'lobby', 'locker_room', 'meeting_room', 'nursery', 'office', 'restaurant',
    'stairscase', 'waitingroom'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4, help="@todo")
    parser.add_argument('--train', action='store_true', default=True, help="@todo")
    parser.add_argument('--num-worker', type=int, default=1, help="@todo")
    parser.add_argument('--arch', type=str, default='regnety_128gf', help="@todo")
    parser.add_argument('--epochs', type=int, default=100, help="@todo")
    parser.add_argument('--resolution', type=int, default=384, help="@todo")
    parser.add_argument('--lr', type=float, default=0.001, help="@todo")
    return parser.parse_known_args()