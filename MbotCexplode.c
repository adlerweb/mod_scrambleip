
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

#include "MbotCexplode.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

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
		printf("Cexplode_getnext(): amnt%d, index%d => next would overflow\n",exp_obj->amnt,exp_obj->index);
#endif
		return NULL;
	}
	exp_obj->index++;
	return exp_obj->strings[exp_obj->index-1];
/*	tmp=Cexplode_getNth(exp_obj->index+1,exp_obj);
	return tmp;
*/
}

char *Cexplode_getbyid(CexplodeStrings *exp_obj, int index) {
	if(NULL==exp_obj) return NULL;
	if(exp_obj->amnt<index) return NULL;
	return exp_obj->strings[index];
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
        perror("Warning, invalid args to Cexplode_removeNth()");
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
		perror("Cexplode_concat realloc FAILED!\n");
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
        perror("Invalid params given to Cexplode!\n");
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
		printf("Malloc Failed at %s:%d tried %d bytes",__FILE__,__LINE__,delimL+1);
		return ECexplodeRet_InternalFailure;
	}
	memcpy(exp_obj->separator,delim,delimL);
	exp_obj->separator[delimL]='\0';
    if(delimL>=stringL)
    {
        printf("Delimiter longer than string => No pieces can be found! (returning original string)\n");
        tmp=malloc(sizeof(char *));
        if(NULL==tmp)
        {
        	perror("Cexplode: Malloc failed!\n");
            return ECexplodeRet_InternalFailure;
        }
        //alloc also for \0
        tmp[0]=malloc(sizeof(char *)*(stringL+1)); 
        if(NULL==tmp[0])
        {
            perror("Cexplode: Malloc failed!\n");
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
                    perror("Cexplode: Malloc failed!\n");
                    return ECexplodeRet_InternalFailure;
                }
                //alloc also for \0
                tmp[pieces-1]=malloc(sizeof(char *)*(index-string_start+1)); 
                if(NULL==tmp[pieces-1])
                {
                    perror("Cexplode: Malloc failed!\n");
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
	        perror("Cexplode: Malloc failed!\n");
	        return ECexplodeRet_InternalFailure;
	    }
	        tmp[pieces-1]=malloc(sizeof(char *)*(index-string_start+1));
	    if(NULL==tmp[pieces-1])
	    {
	        perror("Cexplode: Malloc failed!\n");
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
		printf("Cexplode_getNth: amnt = 0\n");
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


