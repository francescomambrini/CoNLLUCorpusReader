from nltk.corpus.reader.util import *
#from nltk.corpus.reader.xmldocs import XMLCorpusReader

from nltk.corpus.reader.api import *
from nltk.tokenize import *
import pyconll

class ConllUCorpusReader(CorpusReader):
    """
    Reader for corpora that consist of MyCapitain-compliant documents.
    Sentences and words can be tokenized using the default tokenizers,
    or by custom tokenizers specified as parameters to the constructor.
    """

    # CorpusView = StreamBackedCorpusView

    def __init__(self, root, fileids,
                 # para_block_reader=read_blankline_block,
                 encoding='utf8'):
        """
        Construct a new citable corpus reader for a set of documents
        located at the given root directory.

        Parameters
        ----------
        root : str
            The root directory for this corpus.
        fileids : str or list
            A list or regexp specifying the fileids in this corpus.

        """

        CorpusReader.__init__(self, root, fileids, encoding)

    def _get_conllu_iter(self, fileid):
        """
        Parameters
        ----------
        fileid: str
            The file identifier of the file to read

        Returns
        -------
        generator : use it to iterate over sentences

        """

        return pyconll.iter_from_file(self._root.join(fileid))

    def _set_fileids(self, fileids):
        """If fileids is None, then return the whole corpus;
        if it is a single file then return a 1-element list

        Returns
        -------
        list
            of file pointer(s)

        """
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]
        return fileids

    def raw(self, fileids=None):
        """Returns the given file(s) as a single string.

        Parameters
        ----------
        fileids : None, list, str, path
            file identifier or pointer. If None, then the whole corpus is returned

        Returns
        -------
        str

        """
        fileids = self._set_fileids(fileids)

        raw_texts = []
        for f in fileids:
            for s in self._get_conllu_iter(f):
                raw_texts.append(s.text)

        return concat(raw_texts)

    def words(self, fileids=None):
        """
        :return: the given file(s) as a list of words
            and punctuation symbols.
        :rtype: list(str)
        """
        fileids = self._set_fileids(fileids)
        sents = self.sents(fileids)
        return [t for t in sents]

    def sents(self, fileids=None):
        """
        :return: the given file(s) as a list of
            pyconll.sentences,
        :rtype: list
        """

        fileids = self._set_fileids(fileids)
        sents = []
        for f in fileids:
            sents.extend(list(self._get_conllu_iter(f)))
        return sents


class CiteCorpusView(StreamBackedCorpusView):
    """
    A specialized corpus view for cts-compliant documents.
    Not implemented yet.

    It may help in case of very large corpora, because MyCapitain can be slow when the cite scheme
    is very fine-grained (e.g. with poetry, where you have a cite element per line).
    """

    def __init__(self, fileid):
        raise NotImplementedError
