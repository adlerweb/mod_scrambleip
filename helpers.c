
/* ******************************************************** */
/*                                                          *
 *      Implementation of php's explode written in C        *
 *      Written by  Maz (2008)                              *
 *      http://maz-programmersdiary.blogspot.com/           *
 *                                                          *
 *      You're free to use this piece of code.              *
 *      You can also modify it freely, but if you           *
 *      improve this, you must write the improved code      *
 *      in comments at:                                     *
 *      http://maz-programmersdiary.blogspot.com/           *
 *      or at:                                              *
 *      http://c-ohjelmoijanajatuksia.blogspot.com/         *
 *      or mail the corrected version to me at              *
 *      Mazziesaccount@gmail.com                            *
 *                                                          *
 *      Revision History:									*
 *      - 0.0.7 28.08.2009/Maz  Added mbot_ll_get_last()    *
 *		- 0.0.6 15.08.2009/Maz  Fixed atomic ops            *
 *      - 0.0.5 10.08.2009/Maz  Added trim functions		*
 * 		- 0.0.4 06.08.2009/Maz  Fixed indexing bugs from	*
 * 								_getnext & _getNth			*
 * 								mantis #0000001				*
 *      -v0.0.3 31.07.2009/Maz  Added Cexplode_concat		*
 *      						(untested)					*
 *		- 0.0.2 27.07.2009/Maz Fixed an Off By One error	*
 *      -v0.0.1 16.09.2008/Maz                              *
 *                                                          */
/* ******************************************************** */
#include "generic.h"
#include "helpers.h"
#include <string.h>

#define DEBUGPRINTS



int Cexplode_nextexists(CexplodeStrings exp_obj)
{
	return !(exp_obj.amnt<exp_obj.index+1);
}

char *Cexplode_getnext(CexplodeStrings *exp_obj)
{
	//char *tmp;
	if(NULL==exp_obj)
		return NULL;
	if(exp_obj->amnt<=exp_obj->index)
	{
#ifdef DEBUGPRINTS
		DPRINTC(PrintComp_gen,"Cexplode_getnext(): amnt%d, index%d => next would overflow\n",exp_obj->amnt,exp_obj->index);
#endif
		return NULL;
	}
	exp_obj->index++;
	return exp_obj->strings[exp_obj->index-1];
/*	tmp=Cexplode_getNth(exp_obj->index+1,exp_obj);
	return tmp;
*/
}
int Cexplode_getAmnt(CexplodeStrings exp_obj)
{
    return exp_obj.amnt;
}
char *Cexplode_removeCurrent(CexplodeStrings *exp_obj)
{
    return Cexplode_removeNth(exp_obj->index+1,exp_obj);
}

char *Cexplode_removeNth(int nro,CexplodeStrings *exp_obj)
{
    char *retval;
    if(CEXPLODE_LAST_ITEM==nro)
        nro=exp_obj->amnt;
    if( NULL==exp_obj || exp_obj->amnt>nro || nro<0 )
    {
        EPRINTC(PrintComp_gen,"Warning, invalid args to Cexplode_removeNth()");
        return NULL;
    }
    if(exp_obj->index>=nro)
        exp_obj->index--;
	if(nro==exp_obj->amnt)
		exp_obj->sepwasatend=1;
    retval=exp_obj->strings[nro-1];
    memmove(&(exp_obj->strings[nro-1]),&(exp_obj->strings[nro]),(exp_obj->amnt-nro)*sizeof(char *));
    exp_obj->amnt--;
    exp_obj->strings[exp_obj->amnt]=NULL;
    return retval;
}

char *Cexplode_getlast(CexplodeStrings *exp_obj)
{
	if(NULL==exp_obj)
		return NULL;
	return Cexplode_getNth(exp_obj->amnt,exp_obj);
}
//TODO: TestThis!
size_t Cexplode_getlentilllast(CexplodeStrings exp_obj)
{
	int i;
	size_t retval=0,seplen;
	if(exp_obj.amnt<2)
		return 0;
	seplen=strlen(exp_obj.separator);
	for
	(
		i=0; 
		i < 
			((exp_obj.sepwasatend)?
			 	exp_obj.amnt:
				exp_obj.amnt-1
			) 
		;i++
	)
	{
		retval+=strlen(exp_obj.strings[i])+seplen;
	}
	//if text started with delim (which was cropped off) => original lenght must be increased by one delimlen
	if(exp_obj.startedWdelim)
		retval+=seplen;
	//remove last seplen to get the place before last delimiter (is this what we want? No.)
//	retval-=seplen;
	return retval;
}
//TODO: test this!!
int Cexplode_sepwasatend(CexplodeStrings exp_obj)
{
	return exp_obj.sepwasatend;
}

int Cexplode_concat(CexplodeStrings *first,CexplodeStrings *second)
{
	size_t cpylen;
	size_t newamnt=first->amnt+second->amnt;
    int i;
	first->strings=realloc(first->strings,newamnt*sizeof(char *));
	if(NULL==first->strings)
	{
		EPRINTC(PrintComp_gen,"Cexplode_concat realloc FAILED!\n");
		return -666;
	}
	for(i=0;i<second->amnt;i++)
	{
		cpylen=strlen(second->strings[i])+1;
		first->strings[first->amnt+i]=malloc(cpylen);
		memcpy(first->strings[first->amnt+i],second->strings[i],cpylen);
	}
	first->amnt=newamnt;
	first->sepwasatend=second->sepwasatend;
	return newamnt;
}


int Cexplode(const char *string,const char *delim, CexplodeStrings *exp_obj )
{
    int stringL = 0;
    int delimL  = 0;
    int index;
    int pieces=0;
    int string_start=0;
    char **tmp=NULL;
	
    //Sanity Checks:
    if(NULL==string || NULL==delim || NULL == exp_obj)
    {
        EPRINTC(PrintComp_gen,"Invalid params given to Cexplode!\n");
        return ECexplodeRet_InvalidParams;
    }
	exp_obj->amnt=exp_obj->index=0;
	exp_obj->sepwasatend=0;
	exp_obj->startedWdelim=0;

    stringL = strlen(string);
    delimL  = strlen(delim);

	exp_obj->separator=malloc(delimL+1);
	if(exp_obj->separator==NULL)
	{
		PPRINTC(PrintComp_gen,"Malloc Failed at %s:%d tried %d bytes",__FILE__,__LINE__,delimL+1);
		return ECexplodeRet_InternalFailure;
	}
	memcpy(exp_obj->separator,delim,delimL);
	exp_obj->separator[delimL]='\0';
    if(delimL>=stringL)
    {
        WPRINTC(PrintComp_gen,"Delimiter longer than string => No pieces can be found! (returning original string)\n");
        tmp=malloc(sizeof(char *));
        if(NULL==tmp)
        {
        	PPRINTC(PrintComp_gen,"Cexplode: Malloc failed!\n");
            return ECexplodeRet_InternalFailure;
        }
        //alloc also for \0
        tmp[0]=malloc(sizeof(char *)*(stringL+1)); 
        if(NULL==tmp[0])
        {
            PPRINTC(PrintComp_gen,"Cexplode: Malloc failed!\n");
            return ECexplodeRet_InternalFailure;
        }
        memcpy(tmp[0],string,stringL+1); 
		exp_obj->amnt=1;
		exp_obj->strings=tmp;
		return 1;
    }

    for(index=0;index<stringL-delimL;index++)
    {
        if(string[index]==delim[0])
        {
            //Check if delim was actually found
            if( !memcmp(&(string[index]),delim,delimL) )
            {
                //token found
                //let's check if token was at the beginning:
                if(index==string_start)
                {
                    string_start+=delimL;
                    index+=delimL-1;
					exp_obj->startedWdelim=1;
                    continue;
                }
                //if token was not at start, then we should add it in CexplodeStrings
                pieces++;   
                if(NULL==tmp)
                    tmp=malloc(sizeof(char *));
                else
                    tmp=realloc(tmp,sizeof(char *)*pieces);
                if(NULL==tmp)
                {
                    PPRINTC(PrintComp_gen,"Cexplode: Malloc failed!\n");
                    return ECexplodeRet_InternalFailure;
                }
                //alloc also for \0
                tmp[pieces-1]=malloc(sizeof(char *)*(index-string_start+1)); 
                if(NULL==tmp[pieces-1])
                {
                    PPRINTC(PrintComp_gen,"Cexplode: Malloc failed!\n");
                    return ECexplodeRet_InternalFailure;
                }
                memcpy(tmp[pieces-1],&(string[string_start]),index-string_start); 

                tmp[pieces-1][index-string_start]='\0'; 
                string_start=index+delimL;
                index+=(delimL-1);
            }//delim found
        }//first letter in delim found from string
    }//for loop

    if(memcmp(&(string[index]),delim,delimL))
	{
    	index+=delimL;
	}
	else
	{
		//Token was last piece in string
		exp_obj->sepwasatend=1;
	}
    if(index!=string_start)
    {
		pieces++;
	    if(NULL==tmp)
	        tmp=malloc(sizeof(char *));
	    else
	        tmp=realloc(tmp,sizeof(char *)*pieces);
	    if(NULL==tmp)
	    {
	        PPRINTC(PrintComp_gen,"Cexplode: Malloc failed!\n");
	        return ECexplodeRet_InternalFailure;
	    }
	        tmp[pieces-1]=malloc(sizeof(char *)*(index-string_start+1));
	    if(NULL==tmp[pieces-1])
	    {
	        PPRINTC(PrintComp_gen,"Cexplode: Malloc failed!\n");
	        return ECexplodeRet_InternalFailure;
	    }
	    memcpy(tmp[pieces-1],&(string[string_start]),index-string_start);
	    tmp[pieces-1][index-string_start]='\0'; //MazFix 1
    }
    exp_obj->amnt=pieces;
    exp_obj->strings=tmp;
    return pieces;
}


char *Cexplode_getNth(int index,CexplodeStrings *_exp_obj)
{
	if(_exp_obj->amnt==0)
	{
#ifdef DEBUGPRINTS
		DPRINTC(PrintComp_gen,"Cexplode_getNth: amnt = 0\n");
#endif
		return NULL;
	}
    if(_exp_obj->amnt<index)
    {
        return NULL;
    }
	_exp_obj->index=index;
    return _exp_obj->strings[index-1];
}

char *Cexplode_getfirst(CexplodeStrings *exp_obj)
{
    return Cexplode_getNth(1,exp_obj);
}

void Cexplode_free_allButPieces(CexplodeStrings exp_obj)
{
    //int i=0;
	
	exp_obj.sepwasatend=0;
	exp_obj.startedWdelim=0;
	exp_obj.index=0;
	if(NULL!=exp_obj.separator)
		free(exp_obj.separator);
	if(NULL!=exp_obj.strings)
    	free(exp_obj.strings);
	exp_obj.amnt=0;
	exp_obj.separator=NULL;
	exp_obj.strings=NULL;
}

void Cexplode_free(CexplodeStrings exp_obj)
{
    int i=0;
	
	exp_obj.sepwasatend=0;
	exp_obj.startedWdelim=0;
	exp_obj.index=0;
    for(;i<exp_obj.amnt;i++)
        free(exp_obj.strings[i]);
	if(NULL!=exp_obj.separator)
		free(exp_obj.separator);
	if(NULL!=exp_obj.strings)
    	free(exp_obj.strings);
	exp_obj.amnt=0;
	exp_obj.separator=NULL;
	exp_obj.strings=NULL;
}


/**
\brief removes all trimchars from a string.
\returns number of characters removed
*/
int mbot_trimall(char *text, char trimchar)
{
    char *tmp;
    size_t len;
    len=strlen(text)+1; //include '\0'
    int retval=0,i;
    
    tmp=malloc(len);
    if(NULL==tmp)
    {
        PPRINTC(PrintComp_gen,"malloc() FAILED at trimall!\n");
        retval=-1;
    }
    else
    {
        for(i=0;i<len;i++)
        {
            if(text[i]!=trimchar) //If trimchar not found, store tmp[i] to location i-previously_removed_chars
                tmp[i-retval]=text[i];
            else            //If trimchar found, increase amount of removed chars.
                retval++;
        }
        if(retval!=0)//if original char had no trimchars -> do not copy
        {
            if(retval==len)//if all chars in string were trimchars...
                text[0]='\0';
            else
                memcpy(text,tmp,len-retval);
        }
        free(tmp);
    }
    return retval;
}
/**
\brief removes trailing trimchars as well as trimchars from the beginning of a string.
\returns number of characters removed
*/
int mbot_lrtrim(char *text, char trimchar)
{
    int tmp;
    tmp=mbot_ltrim(text,trimchar);
    tmp+=mbot_rtrim(text,trimchar);
    return tmp;
}
/**
\brief removes trimchars from the beginning of a string.
\returns number of characters removed
*/
int mbot_ltrim(char *text, char trimchar)
{
    size_t i,len;
    len=strlen(text);
    for(i=0;i<len && trimchar==text[i];i++)
    {
        DPRINTC(PrintComp_gen,"trimchar found at index %d\n",i);
    }
    DPRINTC(PrintComp_gen,"Trim done, %d trimchars found\n",i);
    if(i>0)
    {
        memmove(text,text+(size_t)i,(len+1-i)); //move. len+1 to include '\0'
    }
    return i;
}

/**
\brief removes trailing trimchars from a string.
\returns number of characters removed
*/
int mbot_rtrim(char *text, char trimchar)
{
    int i;
    size_t len;
    int retval=0;
    len=strlen(text);
    for
    (
      i=len-1;
      i>=0&&( text[i]==trimchar );
      i--
    )
    {
        text[i]=(char)0;
        retval++;
    }
    return retval;
}
void MbotAtomic32Uninit(MbotAtomic32 **_this_)
{
	/* This is not really safe - destroying a semaphore someone is waiting may result undefined behaviour... */
	sem_destroy(&((*_this_)->sem));
	free(*(_this_));
	_this_=NULL;
}

MbotAtomic32 * MbotAtomic32Init()
{
	MbotAtomic32 *_this;
	_this=malloc(sizeof(MbotAtomic32));
	if(NULL==_this)
	{
		PPRINTC(PrintComp_gen,"OMG! malloc returned NULL!!! at %s:%d\n",__FILE__,__LINE__);
		return NULL;
	}
	_this->value=0;
	if(-1==sem_init(&_this->sem,0,1))
	{
		perror("OMG! sem_init failed\n");
		free(_this);
		return NULL;
	}
	return _this;
}

#ifndef ARCH_x86
unsigned int mbot_atomicCAS(MbotAtomic32* atomic, unsigned int old, unsigned int newval)
{
	unsigned int retval;
	sem_wait(&atomic->sem);
	retval=(atomic->value);
	if(atomic->value==old)
		atomic->value=newval;
	sem_post(&atomic->sem);
	return retval;
}

unsigned int mbot_atomicSet(MbotAtomic32* atomic,unsigned int value)
{
	int tmp;
	sem_wait(&atomic->sem);
	tmp=atomic->value=value;
	sem_post(&atomic->sem);
    return tmp;
}


#else
unsigned int mbot_atomicSet(MbotAtomic32* atomic,unsigned int value)
{
    return atomic->value=value;
}
#endif
unsigned int mbot_atomicGet(MbotAtomic32* atomic)
{
    return atomic->value;
}
unsigned int mbot_atomicDecIfGreater(MbotAtomic32* atomic,unsigned int decrement, unsigned int cmp)
{
	unsigned int tmp;
	while(1)
	{
		//get old value
		tmp=atomic->value;
		if(tmp<=cmp)
			break;
		//try setting new -> if success -> atomic dec done, else try again.
		if(tmp==mbot_atomicCAS(atomic,tmp,tmp-decrement))
		{
			break;
		}
	}
	return tmp;
}
unsigned int mbot_atomicIncIfNequal(MbotAtomic32* atomic,unsigned int addition, unsigned int cmp)
{
	unsigned int tmp;
	while(1)
	{
		//get old value
		tmp=atomic->value;
		if(tmp==cmp)
			break;
		//try setting new -> if success -> atomic inc done, else try again.
		if(tmp==mbot_atomicCAS(atomic,tmp,tmp+addition))
		{
			break;
		}
	}
	return tmp;
}
unsigned int mbot_atomicIncIfGreater(MbotAtomic32* atomic,unsigned int addition, unsigned int cmp)
{
	unsigned int tmp;
	while(1)
	{
		//get old value
		tmp=atomic->value;
		if(tmp<=cmp)
			break;
		//try setting new -> if success -> atomic inc done, else try again.
		if(tmp==mbot_atomicCAS(atomic,tmp,tmp+addition))
		{
			break;
		}
	}
	return tmp;
}
unsigned int mbot_atomicIncIfSmaller(MbotAtomic32* atomic,unsigned int addition, unsigned int cmp)
{
	unsigned int tmp;
	while(1)
	{
		//get old value
		tmp=atomic->value;
		if(tmp>=cmp)
			break;
		//try setting new -> if success -> atomic inc done, else try again.
		if(tmp==mbot_atomicCAS(atomic,tmp,tmp+addition))
		{
			break;
		}
	}
	return tmp;
}
unsigned int mbot_atomicIncIfEqual(MbotAtomic32* atomic,unsigned int decrement, unsigned int cmp)
{
	unsigned int tmp;
	while(1)
	{
		//get old value
		tmp=atomic->value;
		if(tmp!=cmp)
			break;
		//try setting new -> if success -> atomic inc done, else try again.
		if(tmp==mbot_atomicCAS(atomic,tmp,tmp+decrement))
		{
			break;
		}
	}
	return tmp;
}
unsigned int mbot_atomicDecIfSmaller(MbotAtomic32* atomic,unsigned int decrement, unsigned int cmp)
{
	unsigned int tmp;
	while(1)
	{
		//get old value
		tmp=atomic->value;
		if(tmp>=cmp)
			break;
		//try setting new -> if success -> atomic inc done, else try again.
		if(tmp==mbot_atomicCAS(atomic,tmp,tmp-decrement))
		{
			break;
		}
	}
	return tmp;
}
unsigned int mbot_atomicDec(MbotAtomic32* atomic,unsigned int decrement)
{
	unsigned int tmp;
	while(1)
	{
		//get old value
		tmp=atomic->value;
		//try setting new -> if success -> atomic inc done, else try again.
		if(tmp==mbot_atomicCAS(atomic,tmp,tmp-decrement))
		{
			break;
		}
	}
	return tmp;
}

unsigned int mbot_atomicAdd(MbotAtomic32* atomic,unsigned int addition)
{
	unsigned int tmp;
	while(1)
	{
		//get old value
		tmp=atomic->value;
		//try setting new -> if success -> atomic inc done, else try again.
		if(tmp==mbot_atomicCAS(atomic,tmp,tmp+addition))
		{
			break;
		}
	}
	return tmp;
}
/* Containers */

mbot_linkedList *mbot_ll_init()
{
    mbot_linkedList* _this;
    _this=malloc(sizeof(mbot_linkedList));
    if(NULL==_this)
    {
        perror("Malloc Failed at linked list Init!");
        return NULL;
    }
    _this->head=_this;
    _this->next=NULL;
    _this->prev=NULL;
    _this->data=NULL;
	return _this;
}
//Warning! This assumes that each "container" in list holds at least itemsize bytes of data - and copies exactly itemsize bytes.
//Usable only for lists which hold fixed size items!
mbot_linkedList *mbot_ll_copylist_wdata(mbot_linkedList *old,size_t itemsize)
{
	mbot_linkedList *newlist;
	newlist=mbot_ll_init();
	if(NULL==newlist)
		return NULL;
	old=mbot_ll_get_next(old);
	while(old!=NULL)
	{
		void *tmp;
		tmp=malloc(itemsize);
		if(NULL==tmp)
		{
			PPRINTC(PrintComp_gen,"PANIC: Malloc FAILED at mbot_ll_copylist_wdata()\n");
			return NULL;
		}
		memcpy(tmp,mbot_ll_dataGet(old),itemsize);
		if(NULL==mbot_ll_add(newlist,tmp))
		{
			EPRINTC(PrintComp_gen,"PANIC: mbot_ll_add() FAILED\n");
			return NULL;
		}
	}
	return newlist;
}

mbot_linkedList * mbot_ll_get_prev(mbot_linkedList *_this)
{
    if(NULL==_this||NULL==_this->head)
    {
        EPRINTC(PrintComp_gen,"ERROR: Null item given to mbot_ll_get_next!\n");
        return NULL;
    }
    if(_this==_this->head || _this==_this->head->next)
    {
        WPRINTC(PrintComp_gen,"WARNING: mbot_ll_get_prev() requested, but first item already given as param!");
        return NULL;   
    }
    return _this->next;
}

mbot_linkedList * mbot_ll_get_next(mbot_linkedList *_this)
{
    if(NULL==_this||NULL==_this->head)
    {
        EPRINTC(PrintComp_gen,"ERROR: Null item given to mbot_ll_get_next!\n");
        return NULL;
    }
    return _this->next;
}
mbot_linkedList * mbot_ll_get_last(mbot_linkedList *_this)
{
    if(NULL==_this||NULL==_this->head)
    {
        EPRINTC(PrintComp_gen,"ERROR: Null item given to mbot_ll_get_first!\n");
        return NULL;
    }   
 	while(NULL!=_this->next)
		_this=_this->next;
	return _this;
}

mbot_linkedList * mbot_ll_get_first(mbot_linkedList *_this)
{
    if(NULL==_this||NULL==_this->head)
    {
        EPRINTC(PrintComp_gen,"ERROR: Null item given to mbot_ll_get_first!\n");
        return NULL;
    }   
    return _this->head->next;
}


mbot_linkedList * mbot_ll_add(mbot_linkedList *_this,void *data)
{
    mbot_linkedList *tmp;
    if(NULL==_this||NULL==_this->head)
    {
        EPRINTC(PrintComp_gen,"ERROR: Null item given to mbot_ll_add!\n");
        return NULL;
    }
    tmp=_this;
    while(NULL!=tmp->next)
    {
        tmp=tmp->next;
    }
    //Set this items next to new item. (Create new item)
    tmp->next=malloc(sizeof(mbot_linkedList));
    if(NULL==tmp->next)
    {
        PPRINTC(PrintComp_gen,"Malloc FAILED at mbot_ll_add()!\n");
        return NULL;
    }
    //Set new item's data to data.
    tmp->next->data=data;
    //set new item's previous to this item.
    tmp->next->prev=tmp;
    //Set new item's next to NULL.
    tmp->next->next=NULL;
    //Set new item's head.
    tmp->next->head=tmp->head;
	return tmp->next;
}

mbot_linkedList * mbot_ll_release(mbot_linkedList *_this)
{
    if(NULL==_this||NULL==_this->head||NULL==_this->prev)
    {
        EPRINTC(PrintComp_gen,"ERROR: Null item given to mbot_ll_release!\n");
        return NULL;
    }
	if(_this==_this->head)
	{
		WPRINTC(PrintComp_gen,"mbot_ll_release(): Thou Shall Not Release The Head!");
		return NULL;
	}
	else
	{
		//Set previous item's next to point next after one to be released
		_this->prev->next=_this->next;
		//Set next item's prev to point the one before released item.
		if(NULL!=_this->next)
			_this->next->prev=_this->prev;
		return _this;
	}
}
mbot_linkedList * mbot_ll_head_get(mbot_linkedList *_this)
{
    if(NULL==_this||NULL==_this->head)
    {
        EPRINTC(PrintComp_gen,"ERROR: Null item given to mbot_ll_release!\n");
        return NULL;
    }
    return _this->head;
}

mbot_linkedList * mbot_ll_safe_release(mbot_linkedList *_this,void *data)
{
    mbot_linkedList *tmp;
    mbot_linkedList *tmp2;
    if(NULL==_this||NULL==_this->head||NULL==data)
    {
        EPRINTC(PrintComp_gen,"ERROR: Null item given to mbot_ll_release!\n");
        return NULL;
    }
    tmp=_this->head;
    while(NULL!=tmp->next)
    {
        if(tmp->next->data==data)
        {
            tmp2=tmp->next;
            if(tmp->next->next!=NULL)
            {
                tmp->next->next->prev=tmp;
            }
            tmp->next=tmp->next->next;
            break;
        }
    }
    return tmp2;
}

void * mbot_ll_dataGet(mbot_linkedList *_this)
{
	if(NULL==_this)
	{
		EPRINTC(PrintComp_gen,"NULL given to mbot_ll_dataGet()!!!\n");
		return NULL;
	}
	return _this->data;
}
void * mbot_ll_dataSet(mbot_linkedList *_this,void *data)
{
	void *tmp;
	if(NULL==_this)
	{
		EPRINTC(PrintComp_gen,"NULL given to mbot_ll_dataGet()!!!\n");
		return NULL;
	}
	tmp=_this->data;
	_this->data=data;
	return tmp;
}
mbot_linkedList * mbot_ll_seek(mbot_linkedList *_this, void *data, size_t datasize)
{
	mbot_linkedList *tmp;
	if(NULL==_this || NULL==_this->head)
	{
		EPRINTC(PrintComp_gen,"WARNING: NULL ptr given to mbot_ll_destroy(), double destroy?");
	}
	tmp=_this->head->next;
	while(NULL!=tmp)
	{
		if(!memcmp(tmp->data,data,datasize))
			break;
		tmp=tmp->next;
	}
	return tmp; //If match found, tmp==matching elem, else tmp==NULL
}
	

void  mbot_ll_destroy(mbot_linkedList **_this)
{
	mbot_linkedList *tmp;
	if(NULL==_this || NULL==*_this)
	{
		EPRINTC(PrintComp_gen,"WARNING: NULL ptr given to mbot_ll_destroy(), double destroy?");
	}
	tmp=(*_this)->head->next;
	if(tmp==NULL&&(*_this)->head!=*_this)
	{
		WPRINTC(PrintComp_gen,"head item's head member does not point to head O_o - BAD_WRONG_MESSED_LOGIC");
	}

	if(NULL==tmp)
	{
		free(*_this);
		*_this=NULL;
	}
	else
	{
		while(NULL!=tmp->next)
		{
			free(tmp->prev);
			tmp=tmp->next;
		}
		free(tmp);
		*_this=NULL;
	}
}

