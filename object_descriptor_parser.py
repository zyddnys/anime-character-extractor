
from collections import defaultdict
import copy
import sys
import os
import urllib.request
import re

from lark import Lark, ParseTree, Tree
from typing import Dict, List, Optional, Set, Tuple

PARSER = None

class SingleObject :
    def __init__(self, object_name: str, tree: ParseTree) -> None :
        self.object_name = object_name
        self.tree = tree

    def match(self, tags: Dict[str, float]) -> bool :
        def match_single_tag(t: ParseTree) :
            tag = str(t.children[0])
            cmp = str(t.children[1])
            prob = float(t.children[2])
            if tag in tags :
                p = tags[tag]
                if cmp == '>' :
                    if p > prob :
                        return True
                elif cmp == '<' :
                    if p < prob :
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

    # def match_and_filter(self, tags: Dict[str, float]) -> Optional[Tuple[str, Dict[str, float]]] :
    #     for obj in self.objects :
    #         if obj.match(tags) :
    #             return obj.object_name, tags # TODO: filter
    #     return None


class RegexMatcher :
    def __init__(self, t: ParseTree) -> None:
        pass

class PostprocessAction :
    def __init__(self, tags) -> None:
        self.tags = set(tags)

    def is_drop(self) -> bool :
        return len(self.tags) == 0

    def added_tags(self) -> Set[str] :
        return self.tags
    
    def update_set(self, tag: str, to_add_tags: Set[str], to_drop_tags: Set[str]) -> Tuple[Set[str], Set[str]] :
        if len(self.tags) == 0 :
            to_add_tags.difference_update([tag])
            to_drop_tags.add(tag)
        else :
            to_add_tags.update(self.tags)
            to_drop_tags.difference_update(self.tags)
        return to_add_tags, to_drop_tags

    def __repr__(self) -> str:
        if self.is_drop() :
            return 'PostprocessAction::Drop'
        else :
            return f'PostprocessAction::Append[{", ".join(self.tags)}]'

def load_tag2cat() :
    if os.path.exists('tag2cat.txt') :
        with open('tag2cat.txt', 'r', encoding = 'utf-8') as fp :
            lines = [s.strip().split(' ') for s in fp.readlines()]
            return {s[0]: s[1] for s in lines if len(s) == 2}
    else :
        print('tag2cat.txt not found, downloading')
        urllib.request.urlretrieve("https://github.com/zyddnys/anime-character-extractor/releases/download/files/tag2cat.txt", "tag2cat.txt")
        return load_tag2cat()

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

class TagPostprocess :
    def __init__(self, t: ParseTree) -> None:
        self.exact_matches: Dict[str, Dict[str, PostprocessAction]] = defaultdict(dict)
        self.category_matches: Dict[str, Dict[str, PostprocessAction]] = defaultdict(dict)
        self.regex_matches: Dict[str, Dict[re.Pattern, PostprocessAction]] = defaultdict(dict)
        self.prob_matches: Dict[str, Dict[str, (float, PostprocessAction)]] = defaultdict(dict)
        self.any_matches: Dict[str, Dict[str, PostprocessAction]] = defaultdict(dict)
        print('Loading danbooru tags')
        self.tag2cat = load_tag2cat()
        def parse_action(t: ParseTree) -> PostprocessAction :
            if isinstance(t, Tree) :
                if len(t.children) == 0 :
                    # drop
                    return PostprocessAction([])
                else :
                    # append multiple tags
                    tags = [str(x) for x in t.children]
                    return PostprocessAction(tags)
            else :
                # append single tags
                return PostprocessAction([str(t)])
        for child in t.children :
            object_to_match = str(child.children[0])
            if isinstance(child.children[1], Tree) :
                # any
                match_rule = str(child.children[1].data)
                assert match_rule == 'postprocess_match_any'
                self.any_matches[object_to_match][compiled] = parse_action(child.children[2])
            else :
                kind = str(child.children[1].type)
                if kind == 'CNAME' :
                    # category match
                    category = shorten_category(str(child.children[1]))
                    self.category_matches[object_to_match][category] = parse_action(child.children[2])
                elif kind == 'TAG' :
                    # exact match
                    tag = str(child.children[1])
                    self.exact_matches[object_to_match][tag] = parse_action(child.children[2])
                elif kind == 'NUMBER' :
                    # prob match
                    prob = float(child.children[1])
                    self.prob_matches[object_to_match]['*'] = (prob, parse_action(child.children[2]))
                elif kind == 'DOUBLE_QUOTED_STRING' :
                    # regex match
                    regex_str = str(child.children[1])[1:-1]
                    compiled = re.compile(regex_str)
                    self.regex_matches[object_to_match][compiled] = parse_action(child.children[2])

    def _get_applied_rules(self, object_name, match_items) :
        if object_name in match_items :
            rules = match_items[object_name] | (match_items['*'] if '*' in match_items else {})
        else :
            rules = match_items['*'] if '*' in match_items else {}
        return rules

    def apply(self, object_name: str, tags: Dict[str, float]) -> Dict[str, float] :
        tags = copy.deepcopy(tags)
        to_add_tags = set()
        to_drop_tags = set()
        # exact match
        rules = self._get_applied_rules(object_name, self.exact_matches)
        for tag, _ in tags.items() :
            if tag in rules :
                to_add_tags, to_drop_tags = rules[tag].update_set(tag, to_add_tags, to_drop_tags)
        # category match
        rules = self._get_applied_rules(object_name, self.category_matches)
        for tag, _ in tags.items() :
            category = self.tag2cat.get(tag, 'gen')
            if category in rules :
                to_add_tags, to_drop_tags = rules[category].update_set(tag, to_add_tags, to_drop_tags)
        # prob match
        rules = self._get_applied_rules(object_name, self.prob_matches)
        for tag, prob in tags.items() :
            for (prob_to_drop, action) in rules.values() :
                if prob < prob_to_drop :
                    to_add_tags, to_drop_tags = action.update_set(tag, to_add_tags, to_drop_tags)
        # regex match
        rules = self._get_applied_rules(object_name, self.regex_matches)
        for tag, _ in tags.items() :
            for pattern, action in rules.items() :
                if pattern.match(tag) :
                    to_add_tags, to_drop_tags = action.update_set(tag, to_add_tags, to_drop_tags)
        # any match
        rules = self._get_applied_rules(object_name, self.any_matches)
        for _, action in rules.items() :
            to_add_tags, to_drop_tags = action.update_set(tag, to_add_tags, to_drop_tags)
        # update tags
        for tag in to_add_tags :
            tags[tag] = 1
        for tag in to_drop_tags :
            del tags[tag]
        return tags

class Configs :
    def __init__(self, tree: ParseTree) -> None:
        self.cfg = {}
        if hasattr(tree.children[0], 'children') :
            for c in tree.children :
                self.cfg[str(c.children[0])] = str(c.children[1][1:-1])
        else :
            self.cfg[str(tree.children[0])] = str(tree.children[1][1:-1])

def verify_descriptor(tree: ParseTree) -> bool :
    return True

def create_objects_from_descriptor(d: str) -> Tuple[Configs, MultipleObjects, TagPostprocess] :
    global PARSER
    if PARSER is None :
        with open('object_descriptor.lark', 'r') as fp :
            PARSER = Lark(fp)
    tree = PARSER.parse(d)
    configs_tree = tree.children[0]
    objects_tree = tree.children[1]
    postprocess_tree = tree.children[2]
    objects: List[SingleObject] = []
    if verify_descriptor(objects_tree) :
        for t in objects_tree.children :
            objects.append(SingleObject(t.children[0], t.children[1]))
    else :
        raise RuntimeError()
    return Configs(configs_tree), MultipleObjects(objects), TagPostprocess(postprocess_tree)

def test() :
    global PARSER
    if PARSER is None :
        with open('object_descriptor.lark', 'r') as fp :
            PARSER = Lark(fp)
    with open('characters\ReincarnatedPrincess.booru', 'r') as fp :
        object_file_str = fp.read()
    configs, objects, postproc = create_objects_from_descriptor(object_file_str)
    obj1 = {
        "1girl": 0.9913652,
        "hair_bow": 0.97715604,
        "purple_eyes": 0.99842465,
        "shirt": 0.7659578,
        "silver_hair": 0.9591773,
        "a_(b)": 0.9,
        "touhou": 1.0
    }
    objname = objects.match(obj1)
    print('matched', objname)
    old_tags = obj1
    new_tags = postproc.apply(objname, old_tags)
    print('old', old_tags)
    print('new', new_tags)

if __name__ == '__main__' :
    test()

