
import sys

from lark import Lark, ParseTree
from typing import Dict, List, Optional, Tuple

PARSER = None

class SingleObject :
    def __init__(self, object_name: str, tree: ParseTree) -> None :
        self.object_name = object_name
        self.tree = tree

    def match(self, tags: Dict[str, float]) -> bool :
        def match_single_tag(t: ParseTree) :
            tag = str(t.children[0])
            prob = float(t.children[1])
            if tag in tags :
                p = tags[tag]
                if p > prob :
                    return True
            return False
        def match_and_query(t: ParseTree) :
            ret = True
            for c in t.children :
                if c.data == 'and_query' :
                    ret = ret and match_and_query(c)
                elif c.data == 'single_tag' :
                    ret = ret and match_single_tag(c)
                elif c.data == 'query' :
                    ret = ret and match_query(c)
            return ret
        def match_query(t: ParseTree) :
            ret = False
            for c in t.children :
                if c.data == 'and_query' :
                    ret = ret or match_and_query(c)
                elif c.data == 'single_tag' :
                    ret = ret or match_single_tag(c)
                elif c.data == 'query' :
                    ret = ret or match_query(c)
            return ret
        if self.tree.data == 'query' :
            return match_query(self.tree)
        elif self.tree.data == 'and_query' :
            return match_and_query(self.tree)
        else :
            raise NotImplemented

class MultipleObjects :
    def __init__(self, objects: List[SingleObject]) -> None:
        self.objects = objects

    def match(self, tags: Dict[str, float]) -> Optional[str] :
        for obj in self.objects :
            if obj.match(tags) :
                return obj.object_name
        return None

def verify_descriptor(tree: ParseTree) -> bool :
    passed = [True]
    def verify2(t: ParseTree) :
        if t.data == 'single_tag' :
            tag = str(t.children[0])
            prob = float(t.children[1])
            if prob < 0.75 :
                passed[0] = False
                print(f'Tag {tag} has threshold set to {prob} which is below the minimum 0.5', file = sys.stderr)
        else :
            for q in t.children :
                verify2(q)
    def verify_object(t: ParseTree) :
        for q in t.children[1:] :
            verify2(q)
    for child in tree.children :
        verify_object(child)
    return passed[0]

def create_objects_from_descriptor(d: str) -> MultipleObjects :
    global PARSER
    if PARSER is None :
        with open('object_descriptor.lark', 'r') as fp :
            PARSER = Lark(fp)
    tree = PARSER.parse(d)
    objects: List[SingleObject] = []
    if verify_descriptor(tree) :
        for t in tree.children :
            objects.append(SingleObject(t.children[0], t.children[1]))
    else :
        raise RuntimeError()
    return MultipleObjects(objects)

def test() :
    global PARSER
    if PARSER is None :
        with open('object_descriptor.lark', 'r') as fp :
            PARSER = Lark(fp)
    with open('extract_ReincarnatedPrincess.txt', 'r') as fp :
        object_file_str = fp.read()
    objects = create_objects_from_descriptor(object_file_str)
    obj1 = {
        "1girl": 0.9913652,
        "hair_bow": 0.97715604,
        "purple_eyes": 0.99842465,
        "shirt": 0.7659578,
        "silver_hair": 0.9591773,
    }
    print(objects.match(obj1))

if __name__ == '__main__' :
    test()

