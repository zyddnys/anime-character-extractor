
from collections import defaultdict
import copy
import sys
import os
import urllib.request
import re

from lark import Lark, ParseTree, Tree, Transformer, v_args
from typing import Dict, List, Optional, Set, Tuple

PARSER = None

class SingleObject :
    def __init__(self, object_name: str, condition) -> None :
        self.object_name = object_name
        self.condition = condition

    def match(self, tags: Dict[str, float]) -> bool :
        def match_single_tag(tag, op, prob) :
            if tag in tags :
                p = tags[tag]
                if op == '>' :
                    if p > prob :
                        return True
                elif op == '<' :
                    if p < prob :
                        return True
            return False
        def handle_condition(kind, args) :
            if kind == 'and' :
                passed = True
                for cdt in args :
                    passed = passed and handle_condition(*cdt)
                return passed
            elif kind == 'or' :
                passed = False
                for cdt in args :
                    passed = passed or handle_condition(*cdt)
                return passed
            elif kind == 'not' :
                return not handle_condition(*args)
            elif kind == 'single_tag' :
                return match_single_tag(*args)
        return handle_condition(*self.condition)
        
class MultipleObjects :
    def __init__(self, objects: List[SingleObject]) -> None:
        self.objects = [obj for obj in objects if obj.object_name != 'PRECONDITION']
        self.precondition = None
        for obj in objects :
            if obj.object_name == 'PRECONDITION' :
                self.precondition = obj

    def match(self, tags: Dict[str, float]) -> Optional[str] :
        for obj in self.objects :
            if obj.match(tags) :
                return obj.object_name
        return None

    def has_precondition(self) -> bool :
        return self.precondition is not None

    def match_precondition(self, tags: Dict[str, float]) -> bool :
        if self.precondition is None :
            return True
        return self.precondition.match(tags)

    # def match_and_filter(self, tags: Dict[str, float]) -> Optional[Tuple[str, Dict[str, float]]] :
    #     for obj in self.objects :
    #         if obj.match(tags) :
    #             return obj.object_name, tags # TODO: filter
    #     return None


class RegexMatcher :
    def __init__(self, t: ParseTree) -> None:
        pass

class PostprocessAction :
    def __init__(self, to_add_tags, to_drop_tags) -> None:
        self.to_add_tags = set(to_add_tags)
        self.to_drop_tags = set(to_drop_tags)
    
    def update_set(self, tag: str, tags: Dict[str, float]) -> Dict[str, float] :
        if len(self.to_add_tags) == 0 and len(self.to_drop_tags) == 0 and tag :
            if tag in tags :
                del tags[tag]
        for t in self.to_drop_tags :
            if t in tags :
                del tags[t]
        for t in self.to_add_tags :
            tags[t] = 1.0
        return tags

    def __repr__(self) -> str:
        if len(self.to_add_tags) == 0 :
            return f'PostprocessAction::Drop[{", ".join(self.to_drop_tags)}]'
        else :
            return f'PostprocessAction::Append[{", ".join(self.to_add_tags)}]'

def load_tag2cat() :
    from utils import download_model_file
    download_model_file('models/tag2cat.txt', "https://github.com/zyddnys/anime-character-extractor/releases/download/files/tag2cat.txt", 'e5d4b2e144d47b044555e8ac72cd4a460d03e36000c5f6c5cc706003d4ada51e')
    with open('models/tag2cat.txt', 'r', encoding = 'utf-8') as fp :
        lines = [s.strip().split(' ') for s in fp.readlines()]
        return {s[0]: s[1] for s in lines if len(s) == 2}

def shorten_category(category: str) :
    if category == 'general' :
        return 'gen'
    elif category == 'character' :
        return 'char'
    elif category == 'artist' :
        return 'art'
    elif category == 'copyright' :
        return 'copy'
    elif category == 'meta' :
        return 'meta'
    else :
        raise Exception(f'Unknown category {category}')

class TagPostprocessCondition :
    def __init__(self, *args, **kwargs) -> None:
        pass

    def apply_to_single_tag(self) -> bool :
        raise NotImplemented
    
    def is_match_single(self, tag: str, category: str, prob: float) -> bool :
        raise NotImplemented
    
    def is_match_all(self, tags: Dict[str, float]) -> bool :
        raise NotImplemented
    
class TagPostprocessConditionRegex(TagPostprocessCondition) :
    def __init__(self, regex) -> None :
        self.r = re.compile(regex)

    def apply_to_single_tag(self) -> bool :
        return True

    def is_match_single(self, tag: str, category: str, prob: float) -> bool :
        return self.r.match(tag) is not None
    
class TagPostprocessConditionCategory(TagPostprocessCondition) :
    def __init__(self, category) -> None :
        self.category = category

    def apply_to_single_tag(self) -> bool :
        return True

    def is_match_single(self, tag: str, category: str, prob: float) -> bool :
        return category == self.category
    
class TagPostprocessConditionProb(TagPostprocessCondition) :
    def __init__(self, prob) -> None :
        self.prob = prob

    def apply_to_single_tag(self) -> bool :
        return True

    def is_match_single(self, tag: str, category: str, prob: float) -> bool :
        return prob <= self.prob
    
class TagPostprocessConditionExact(TagPostprocessCondition) :
    def __init__(self, tag) -> None :
        self.tag = tag

    def apply_to_single_tag(self) -> bool :
        return True

    def is_match_single(self, tag: str, category: str, prob: float) -> bool :
        return tag == self.tag
    
class TagPostprocessConditionTag(TagPostprocessCondition) :
    def __init__(self, condition) -> None :
        self.condition = SingleObject('', condition)

    def apply_to_single_tag(self) -> bool :
        return False

    def is_match_all(self, tags: Dict[str, float]) -> bool :
        return self.condition.match(tags)
    
class TagPostprocessConditionAny(TagPostprocessCondition) :
    def __init__(self) -> None :
        pass

    def apply_to_single_tag(self) -> bool :
        return False

    def is_match_all(self, tags: Dict[str, float]) -> bool :
        return True
    
class TagPostprocessEntry :
    def __init__(self, object_name: str, condition: TagPostprocessCondition, action: PostprocessAction) -> None:
        self.object_name = object_name
        self.condition = condition
        self.action = action

class TagPostprocess :
    def __init__(self, entries: List[TagPostprocessEntry]) -> None:
        self.entries = entries
        print('Loading danbooru tags')
        self.tag2cat = load_tag2cat()

    def apply(self, object_name: str, tags: Dict[str, float]) -> Dict[str, float] :
        for e in self.entries :
            if e.object_name == '*' or object_name == e.object_name :
                if e.condition.apply_to_single_tag() :
                    tags2 = copy.deepcopy(tags)
                    for (tag, prob) in tags2.items() :
                        category = self.tag2cat.get(tag, 'gen')
                        if e.condition.is_match_single(tag, category, prob) :
                            tags = e.action.update_set(tag, tags)
                else :
                    if e.condition.is_match_all(tags) :
                        tags = e.action.update_set(None, tags)
        return tags

class Configs :
    def __init__(self, cfg) -> None:
        self.cfg = cfg

def verify_descriptor(tree: ParseTree) -> bool :
    return True

class TagQueryTransformer(Transformer) :
    @v_args(inline = True)
    def single_tag(self, tag, compare, number):
        return ('single_tag', (tag, str(compare), number))
    
    @v_args(inline = True)
    def not_expr(self, not_symbol, single_tag) :
        return ('not', single_tag)
    
    def or_expr(self, args) :
        return ('or', list(args))

    def and_expr(self, args) :
        return ('and', list(args))
    
    @v_args(inline = True)
    def object(self, obj_name, conditions) :
        return (str(obj_name), conditions)
    
    @v_args(inline = True)
    def postprocess_action_drop(self, drop_symbol, *tags) :
        return PostprocessAction([], [str(x) for x in tags])
    
    @v_args(inline = True)
    def postprocess(self, object_name, condition, action) :
        if isinstance(condition, Tree) and str(condition.data) == 'any_symbol' :
            condition = TagPostprocessConditionAny()
        if isinstance(action, Tree) and str(action.data) == 'drop_symbol' :
            action = PostprocessAction([], [])
        return TagPostprocessEntry(object_name, condition, action)

    @v_args(inline = True)
    def postprocess_action_append(self, append_symbol, *tags) :
        return PostprocessAction([str(x) for x in tags], [])

    @v_args(inline = True)
    def postprocess_match_regex(self, regex_symbol, regex) :
        return TagPostprocessConditionRegex(regex)
    
    @v_args(inline = True)
    def postprocess_match_category(self, category_symbol, category) :
        return TagPostprocessConditionCategory(category)

    @v_args(inline = True)
    def postprocess_match_prob(self, prob_symbol, prob) :
        return TagPostprocessConditionProb(prob)

    @v_args(inline = True)
    def postprocess_match_exact(self, exact_symbol, tag) :
        return TagPostprocessConditionExact(tag)
    
    @v_args(inline = True)
    def postprocess_match_tag(self, tag_symbol, condition) :
        return TagPostprocessConditionTag(condition)
        
    def postprocess_statement(self, entries) :
        return TagPostprocess(entries)

    @v_args(inline = True)
    def config(self, key, value) :
        return {key: value}

    def config_statement(self, args) :
        d = {}
        for x in args :
            d.update(x)
        return d

    def DOUBLE_QUOTED_STRING(self, s) :
        return s[1:-1]

    def TAG(self, t) :
        return str(t)
    
    def CNAME(self, t) :
        return str(t)
    
    def NUMBER(self, n) :
        return float(n)

def create_objects_from_descriptor(d: str) -> Tuple[Configs, MultipleObjects, TagPostprocess] :
    global PARSER
    if PARSER is None :
        with open('object_descriptor.lark', 'r') as fp :
            PARSER = Lark(fp, parser = 'lalr', transformer = TagQueryTransformer())
    tree = PARSER.parse(d)
    configs_tree = tree.children[0]
    objects_tree = tree.children[1]
    postprocess_tree = tree.children[2]
    objects: List[SingleObject] = []
    if verify_descriptor(objects_tree) :
        star_object = None
        if isinstance(objects_tree, tuple) :
            (objname, condition) = objects_tree
            if objname == '*' :
                star_object = condition
            else :
                objects.append(SingleObject(objname, condition))
        else :
            for (objname, condition) in objects_tree.children :
                if objname == '*' :
                    star_object = condition
                else :
                    objects.append(SingleObject(objname, condition))
        if len(objects) == 0 :
            raise Exception('Need at least one object')
        if star_object is not None :
            for i in range(len(objects)) :
                objects[i].condition = ('and', [star_object, objects[i].condition])
    else :
        raise RuntimeError()
    return Configs(configs_tree), MultipleObjects(objects), postprocess_tree

def test() :
    global PARSER
    if PARSER is None :
        with open('object_descriptor.lark', 'r') as fp :
            PARSER = Lark(fp, parser = 'lalr', transformer = TagQueryTransformer())
    with open('characters/konosuba.booru', 'r') as fp :
        object_file_str = fp.read()
    configs, objects, postproc = create_objects_from_descriptor(object_file_str)
    obj1 = {'1girl': 0.9995528, 'solo': 0.99735606, 'long_hair': 0.98722327, 'breasts': 0.65833545, 'blush': 0.90023476, 'smile': 0.9793658, 'looking_at_viewer': 0.9446239, 'blue_eyes': 0.98489827, 'blonde_hair': 0.9977075, 'skirt': 0.58093333, 'large_breasts': 0.45485604, 'bangs': 0.68994427, 'hair_ornament': 0.8278184, 'simple_background': 0.83087015, 'dress': 0.93080366, 'shirt': 0.3509593, 'long_sleeves': 0.90752345, 'eyebrows_visible_through_hair': 0.79850316, 'medium_breasts': 0.40998083, 'very_long_hair': 0.4000318, 'hair_between_eyes': 0.57388747, 'standing': 0.36010072, 'braid': 0.96747386, 'closed_mouth': 0.6984893, 'food': 0.55138767, 'upper_body': 0.73705965, 'frills': 0.6462211, 'alternate_costume': 0.7812935, 'looking_back': 0.52319884, 'from_behind': 0.4070084, 'apron': 0.97278893, 'white_dress': 0.9071166, 'from_side': 0.51697314, 'single_braid': 0.73256063, 'fruit': 0.63991135, 'gradient': 0.48306903, 'gradient_background': 0.5170844, 'leaf': 0.48402834, 'no_headwear': 0.77966315, 'no_hat': 0.7410354, 'waist_apron': 0.5234399, 'side_braid': 0.3765417, 'yellow_background': 0.879439, 'maid_apron': 0.45189977, 'white_apron': 0.83162624, 'frilled_apron': 0.6901333, 'skirt_hold': 0.5131409}
    
    old_tags = copy.deepcopy(obj1)
    objname = objects.match(obj1)
    print('matched', objname)
    print('old', old_tags)
    new_tags = postproc.apply(objname, old_tags)
    print('new', new_tags)

if __name__ == '__main__' :
    test()

