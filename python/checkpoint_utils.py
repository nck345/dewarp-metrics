import csv
import json
import os
import subprocess
import threading
import time
import pandas as pd

Drv_lock=threading.Lock()

def is_endpoint_error(err):
	if isinstance(err,OSError) and getattr(err,'errno',None)==107: return True
	s=str(err).lower()
	return '107' in s or 'transport endpoint' in s or 'endpoint is not connected' in s

def remount_google_drive(mnt_dir='/content/drive',max_try=3,gap_sec=2.0):
	with Drv_lock:
		my_drv=os.path.join(mnt_dir,'MyDrive')
		if os.path.exists(my_drv): return True
		try:
			from google.colab import drive
			print(f"⚠️ Transport endpoint disconnected. Remounting {mnt_dir}...")
			for att in range(1,max_try+1):
				try:
					subprocess.run(['fusermount','-u','-z',mnt_dir],capture_output=True)
					subprocess.run(['umount','-l',mnt_dir],capture_output=True)
				except Exception:
					pass
				try:
					drive.mount(mnt_dir,force_remount=True)
					time.sleep(gap_sec)
					if os.path.exists(my_drv):
						print("✅ Remounted Google Drive!")
						return True
				except Exception as ex:
					print(f"   Remount {att}/{max_try} failed: {ex}")
					if att<max_try: time.sleep(gap_sec*(2**(att-1)))
		except ImportError:
			pass
		except Exception as ex:
			print(f"⚠️ Remount error: {ex}")
		return False

def _ensure_drv(path):
	if '/content/drive' in path and not os.path.exists('/content/drive/MyDrive'):
		remount_google_drive()

def safe_to_csv(df,path,max_try=3,**kw):
	abs_p=os.path.abspath(path)
	tmp_p=f"{abs_p}.tmp_{os.getpid()}_{threading.get_ident()}"
	for att in range(1,max_try+1):
		try:
			_ensure_drv(abs_p)
			os.makedirs(os.path.dirname(abs_p),exist_ok=True)
			df.to_csv(tmp_p,**kw)
			os.replace(tmp_p,abs_p)
			return
		except OSError as err:
			if os.path.exists(tmp_p):
				try:
					os.remove(tmp_p)
				except Exception:
					pass
			if is_endpoint_error(err):
				print(f"⚠️ [Errno 107] Saving {path} ({att}/{max_try}). Remounting...")
				remount_google_drive()
				time.sleep(1.0*(2**(att-1)))
			else:
				if att==max_try: raise
				time.sleep(0.5)

def safe_json_dump(obj,path,max_try=3):
	abs_p=os.path.abspath(path)
	tmp_p=f"{abs_p}.tmp_{os.getpid()}_{threading.get_ident()}"
	for att in range(1,max_try+1):
		try:
			_ensure_drv(abs_p)
			os.makedirs(os.path.dirname(abs_p),exist_ok=True)
			with open(tmp_p,'w',encoding='utf-8') as f:
				json.dump(obj,f,ensure_ascii=False,indent=2)
			os.replace(tmp_p,abs_p)
			return True
		except OSError as err:
			if os.path.exists(tmp_p):
				try:
					os.remove(tmp_p)
				except Exception:
					pass
			if is_endpoint_error(err):
				remount_google_drive()
				time.sleep(1.0*(2**(att-1)))
			elif att==max_try:
				return False
			else:
				time.sleep(0.3)
	return False

def safe_read_csv(path,max_try=3):
	for att in range(1,max_try+1):
		try:
			_ensure_drv(path)
			if not os.path.exists(path): return None
			return pd.read_csv(path)
		except OSError as err:
			if is_endpoint_error(err):
				remount_google_drive()
				time.sleep(1.0*(2**(att-1)))
			else:
				return None
		except Exception:
			return None
	return None

class StepCkpt:
	def __init__(self,drv_p,cols,loc_p=None):
		self.drv_p=drv_p
		self.loc_p=loc_p
		self.Cols=list(cols)
		self.Map={}
		self.Lck=threading.Lock()

	def clear(self):
		self.Map.clear()
		for p in (self.drv_p,self.loc_p):
			if p and os.path.exists(p):
				try:
					os.remove(p)
				except Exception:
					pass

	def _parse_df(self,df,val_cols):
		Out={}
		if df is None or df.empty: return Out
		if 'Dewarped_Image' not in df.columns or 'Scan_GT' not in df.columns: return Out
		for _,r in df.iterrows():
			p_nm=r.get('Dewarped_Image')
			s_nm=r.get('Scan_GT')
			if pd.isna(p_nm) or pd.isna(s_nm): continue
			p_str=str(p_nm).strip()
			s_str=str(s_nm).strip()
			if not p_str or not s_str or p_str=='All': continue
			if val_cols:
				ok=True
				for vc in val_cols:
					if vc not in df.columns or pd.isna(r.get(vc)):
						ok=False
						break
				if not ok: continue
			row={}
			for c in self.Cols:
				v=r.get(c,None)
				row[c]=None if pd.isna(v) else v
			row['Dewarped_Image']=p_str
			row['Scan_GT']=s_str
			Out[(p_str,s_str)]=row
		return Out

	def load(self,use_res=True,seed_df=None,val_cols=None):
		with self.Lck:
			if not use_res:
				self.clear()
				return self.Map
			self.Map.clear()
			drv_df=safe_read_csv(self.drv_p) if self.drv_p else None
			loc_df=safe_read_csv(self.loc_p) if self.loc_p else None
			drv_map=self._parse_df(drv_df,val_cols)
			loc_map=self._parse_df(loc_df,val_cols)
			self.Map.update(drv_map)
			need_sync=False
			for k,v in loc_map.items():
				if k not in self.Map:
					self.Map[k]=v
					need_sync=True
			if seed_df is not None and not seed_df.empty:
				seed_map=self._parse_df(seed_df,val_cols)
				for k,v in seed_map.items():
					if k not in self.Map:
						self.Map[k]=v
						need_sync=True
			if need_sync and self.drv_p and self.Map:
				self._dump_path(self.drv_p)
			if self.loc_p and self.Map and (need_sync or len(loc_map)<len(self.Map)):
				self._dump_path(self.loc_p)
			return dict(self.Map)

	def _dump_path(self,path):
		if not path: return
		rows=list(self.Map.values())
		df=pd.DataFrame(rows,columns=self.Cols) if rows else pd.DataFrame(columns=self.Cols)
		safe_to_csv(df,path,index=False,encoding='utf-8')

	def _append_file(self,path,row,max_try=3):
		if not path: return False
		vals=[row.get(c,'') for c in self.Cols]
		for att in range(1,max_try+1):
			try:
				_ensure_drv(path)
				par=os.path.dirname(os.path.abspath(path))
				if par: os.makedirs(par,exist_ok=True)
				need_hdr=(not os.path.exists(path)) or os.path.getsize(path)==0
				with open(path,'a',newline='',encoding='utf-8') as f:
					w=csv.writer(f)
					if need_hdr: w.writerow(self.Cols)
					w.writerow(vals)
				return True
			except OSError as err:
				if is_endpoint_error(err):
					remount_google_drive()
					try:
						self._dump_path(path)
						return True
					except Exception:
						pass
				if att<max_try: time.sleep(0.5*(2**(att-1)))
		return False

	def append(self,row):
		with self.Lck:
			k=(str(row['Dewarped_Image']),str(row['Scan_GT']))
			clean={c:row.get(c,None) for c in self.Cols}
			clean['Dewarped_Image']=k[0]
			clean['Scan_GT']=k[1]
			self.Map[k]=clean
			if self.loc_p: self._append_file(self.loc_p,clean)
			if self.drv_p: self._append_file(self.drv_p,clean)

	def to_df(self,pairs=None):
		with self.Lck:
			if pairs is not None:
				rows=[self.Map[(p[0],p[2])] for p in pairs if (p[0],p[2]) in self.Map]
			else:
				rows=list(self.Map.values())
			if not rows: return pd.DataFrame(columns=self.Cols)
			return pd.DataFrame(rows,columns=self.Cols)
