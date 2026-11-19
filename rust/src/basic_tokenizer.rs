use std::collections::BTreeMap;
pub struct BasicTokenizer { merges: BTreeMap<(usize,usize),usize>, vocab: Vec<Vec<u8>> }
impl BasicTokenizer {
    pub fn new() -> Self { Self { merges: BTreeMap::new(), vocab: (0..256).map(|i| vec![i as u8]).collect() } }
    fn merge(ids: &[usize], pair: (usize,usize), token: usize) -> Vec<usize> {
        let mut out=Vec::new(); let mut i=0;
        while i<ids.len() { if i+1<ids.len() && (ids[i],ids[i+1])==pair {out.push(token);i+=2;} else {out.push(ids[i]);i+=1;} } out
    }
    pub fn train(&mut self,text:&str,size:usize) -> Result<(),String> {
        if text.is_empty() || size<256 {return Err("Nonempty corpus and vocabulary >=256 required".into());}
        *self=Self::new(); let mut ids:Vec<usize>=text.bytes().map(usize::from).collect();
        for token in 256..size {
            let mut counts=BTreeMap::new(); for p in ids.windows(2) {*counts.entry((p[0],p[1])).or_insert(0usize)+=1;}
            let pair=counts.iter().min_by_key(|(p,n)| (std::cmp::Reverse(**n),**p)).map(|(p,_)|*p);
            let Some(pair)=pair else {break};
            let mut bytes=self.vocab[pair.0].clone();bytes.extend(&self.vocab[pair.1]);self.vocab.push(bytes);
            self.merges.insert(pair,token);ids=Self::merge(&ids,pair,token);
        } Ok(())
    }
    pub fn encoder(&self,text:&str)->Vec<usize> {
        let mut ids:Vec<usize>=text.bytes().map(usize::from).collect();
        loop {let pair=ids.windows(2).map(|p|(p[0],p[1])).filter(|p|self.merges.contains_key(p)).min_by_key(|p|self.merges[p]);
            let Some(pair)=pair else {break};ids=Self::merge(&ids,pair,self.merges[&pair]);} ids
    }
    pub fn decoder(&self,ids:Vec<usize>)->Result<String,String> {
        let mut bytes=Vec::new();for id in ids {bytes.extend(self.vocab.get(id).ok_or("Unknown token ID")?);}
        String::from_utf8(bytes).map_err(|e|e.to_string())
    }
}
#[cfg(test)] mod tests { use super::*;
#[test] fn merges_and_roundtrip(){let mut t=BasicTokenizer::new();t.train("abababab",258).unwrap();assert_eq!(t.encoder("abababab"),vec![257,257]);let s="🦉 中文\n";assert_eq!(t.decoder(t.encoder(s)).unwrap(),s);assert!(t.decoder(vec![99999]).is_err());t.train("a",300).unwrap();assert_eq!(t.encoder("ab"),vec![97,98]);}
}
