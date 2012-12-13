
/* **********************************************************************/
/*                                                          			*
 *      Implementation of php's explode written in C        			*
 *      Written by  Maz (2008)                              			*
 *      http://maz-programmersdiary.blogspot.com/           			*
 *                                                          			*
 *      You're free to use this piece of code.              			*
 *      You can also modify it freely, but if you           			*
 *      improve this, you must write the improved code      			*
 *      in comments at:                                     			*
 *      http://maz-programmersdiary.blogspot.com/           			*
 *      or at:                                              			*
 *      http://c-ohjelmoijanajatuksia.blogspot.com/         			*
 *      or mail the corrected version to me at              			*
 *      Mazziesaccount@gmail.com                            			*
 *                                                          			*
 *      Revision History:                                   			*
 *																		*
 * 		- 0.0.6 20.09.2011/Maz  Fixed atomic CAS again						*
 * 		- 0.0.6 15.08.2009/Maz  Fixed atomic CAS						*
 * 		- 0.0.5 11.08.2009/Maz  Added Cexplode_free_allButPieces		*
 *  	- 0.0.4 11.08.2009/Maz  Added atomic ops and 					*
 *  							mbot_ll									*
 *      -v0.0.3 31.07.2009/Maz  Added Cexplode_concat					*
 *      						(untested)								*
 *  	-v0.0.2 21.07.2009/Maz  Some additions for better   			*
 *  							usability in MazBotV4					*
 *      -v0.0.1 16.09.2008/Maz                              			*
 *                                                          			*/
/* ******************************************************************** */

#ifndef HELPERS_H
#define HELPERS_H

/* Some Cexplode calls support using this special item define */
#define CEXPLODE_LAST_ITEM 0xFFFFFFFF

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <semaphore.h>

/**
 * @brief Struct for 32bit wide integer type used in atomic operations
 */
typedef struct MbotAtomic32 
{
	volatile unsigned int value;
	sem_t sem;		///< If non x86 arch is used, these atomic ops are dummies using semaphore
}MbotAtomic32;


/**
 * @brief Struct for Cexplode object
 */
typedef struct CexplodeStrings
{
    int amnt;
    char **strings;
	char *separator;
	int sepwasatend;
	int startedWdelim;
	int index;
}CexplodeStrings;

/**
 * @brief enumeration for Cexplodei's error return values
 */
typedef enum ECexplodeRet
{
    ECexplodeRet_InternalFailure    = -666,
    ECexplodeRet_InvalidParams         = -667
}ECexplodeRet;

/**
* @brief Removes the previously returned piece
*
* Must not be called before calling Cexplode 
* If removed item is last piece, the "sepwasatend" flag will be set true
*
* @param CexplodeStrings *exp_obj pointer to CexplodeStrings type object, filled by call to Cexplode()
* @return ptr to string being removed at success, NULL at failure 
* @see Cexplode, Cexplode_removeNth, Cexplode_getAmnt, Cexplode_nextexists
*/
char *Cexplode_removeCurrent(CexplodeStrings *exp_obj);

/**
 * @brief Removes Nth piece from cexplode 
 * Must not be called before calling Cexplode 
 * If removed item is last piece, the "sepwasatend" flag will be set true! 
 * Note, you can use special CEXPLODE_LAST_ITEM define to remove the last item 
 *
 * @param int nro number of exploded piece to be removed from the CexplodeStrings containing results
 * @param CexplodeStrings *exp_obj pointer to CexplodeStrings type object, filled by call to Cexplode()
 * @return ptr to removed string 
 * @see Cexplode, Cexplode_removeCurrent, Cexplode_getAmnt, Cexplode_nextexists
 */
char *Cexplode_removeNth(int nro,CexplodeStrings *exp_obj);

/**
 * @brief Get the amount of pieces in exploded object 
 * Must not be called before calling Cexplode
 *
 * @param CexplodeStrings *exp_obj pointer to CexplodeStrings type object, filled by call to Cexplode()
 * @return amount of exploded pieces stored in CexplodeStrings container
 * @see Cexplode
 */
int Cexplode_getAmnt(CexplodeStrings exp_obj);

/**
 * @brief Explodes string to pieces according to delimiter. Result is stored in exp_obj and can be retrieved using functions below 
 * The results of explosion are stored in same order as they occurred in initial string, eg. if string "1 2 3 4" 
 * would be exploded with space (" ") as delimiter, Cexplode_getfirst() would return 1, Cexplode_getNth() with n being 4, would return 4.
 *
 * @param const char *string pointer to C string being exploded
 * @param const char *delim pointer to C string used as delimiter for cutting original string
 * @param CexplodeStrings *exp_obj pointer to CexplodeStrings type object, which will be filled to contain results of explosion.
 * @return amount of pieces - number smaller than 1 if an error occurs 
 * @see CexplodeStrings, Cexplode_removeCurrent, Cexplode_removeNth, Cexplode_getAmnt, Cexplode_nextexists, Cexplode_getNth, Cexplode_getfirst, Cexplode_getnext, Cexplode_getlast, Cexplode_free, Cexplode_free_allButPieces, Cexplode_getlentilllast, Cexplode_sepwasatend, Cexplode_concat
 *
 */
int Cexplode(const char *string,const char *delim, CexplodeStrings *exp_obj );

/**
 * @brief Peeks if there's another result in exp_obj. 
 * Must not be called before calling Cexplode 
 *
 * @param CexplodeStrings exp_obj CexplodeStrings type object, filled by call to Cexplode()
 * @return 1 if next piece exists (Eg. if Cexplode_getnext et al. can be safely used), 0 if there's no next result in object.
 * @see Cexplode, Cexplode_getnext
 * */ 
int Cexplode_nextexists(CexplodeStrings exp_obj);

/**
 * @brief Retrieve's Nth exploded piece - first is first (index starts from 1, not from 0)
 * Updates internal iterator, IE following call to Cexplode_getnext will retrieve index+1th piece
 * @param int index index number of result to be retrieved. first is first (index starts from 1, not from 0)
 * @param CexplodeStrings *exp_obj pointer to CexplodeStrings type object, filled by call to Cexplode()
 * @return NULL on error, othervice a pointer to result stored in Cexplode object
 * @warning Must not be called before calling Cexplode 
 * @see Cexplode, Cexplode_getfirst, Cexplode_getnext, Cexplode_getlast, Cexplode_getAmnt
 */
char *Cexplode_getNth(int index,CexplodeStrings *exp_obj);

/**
 * @brief Get's the first exploded piece. Same as Cexplode_getNth(1,*exp_obj);
 * @param CexplodeStrings *exp_obj pointer to CexplodeStrings type object, filled by call to Cexplode()
 * @return NULL on error, othervice a pointer to result stored in Cexplode object
 * @warning Must not be called before calling Cexplode
 * @see Cexplode, Cexplode_getNth, Cexplode_getnext, Cexplode_getlast
 */
char *Cexplode_getfirst(CexplodeStrings *exp_obj);

/**
 * @brief Get's next piece. Returns NULL if no more pieces are around 
 * @param CexplodeStrings *exp_obj pointer to CexplodeStrings type object, filled by call to Cexplode()
 * @return NULL on error, othervice a pointer to result stored in Cexplode object
 * @warning Must not be called before calling Cexplode 
 * @see Cexplode, Cexplode_getNth, Cexplode_getfirst, Cexplode_getlast
 */
char *Cexplode_getnext(CexplodeStrings *exp_obj);
/**
 * @brief Gets last exploded piece 
 * @param CexplodeStrings *exp_obj pointer to CexplodeStrings type object, filled by call to Cexplode()
 * @return NULL on error, othervice a pointer to result stored in Cexplode object
 * @warning Must not be called before calling Cexplode
 * @see Cexplode, Cexplode_getNth, Cexplode_getnext, Cexplode_getfirst
 */
char *Cexplode_getlast(CexplodeStrings *exp_obj);

/**
 * @brief Frees resources allocated by call to Cexplode() - BEWARE frees also splitted pieces
 * @param CexplodeStrings exp_obj CexplodeStrings type object, filled by call to Cexplode()
 * @warning Must not be called before calling Cexplode 
 * @warning BEWARE frees also splitted pieces, in which the returned pointers by Cexplode_get* points.
 * @see Cexplode_free_allButPieces, Cexplode, Cexplode_getNth, Cexplode_getnext, Cexplode_getfirst, Cexplode_getlast
 * */
void Cexplode_free(CexplodeStrings exp_obj);

/**
 * @brief Frees resources allocated by call to Cexplode() - does not free splitted pieces
 * @param CexplodeStrings exp_obj CexplodeStrings type object, filled by call to Cexplode()
 * @warning Must not be called before calling Cexplode
 * @see Cexplode_free, Cexplode, Cexplode_getNth, Cexplode_getnext, Cexplode_getfirst, Cexplode_getlast
 */
void Cexplode_free_allButPieces(CexplodeStrings exp_obj);

/**
 * @brief Gets the amount of chars from the start of the original string to the beginning of last found delimiter
 * @param CexplodeStrings exp_obj CexplodeStrings type object, filled by call to Cexplode()
 * @return amount of chars from the start of the original string to the beginning of last found delimiter
 * @warning Must not be called before calling Cexplode 
 * @see Cexplode, Cexplode_sepwasatend
 * */
size_t Cexplode_getlentilllast(CexplodeStrings exp_obj);

/**
 * @brief returns 1 if last chars in original string were the separator - else returns 0
 * @param CexplodeStrings exp_obj CexplodeStrings type object, filled by call to Cexplode()
 * @return 1 if last chars in original string were the separator - else returns 0
 * @warning Must not be called before calling Cexplode 
 * @see Cexplode, Cexplode_getlentilllast
 */
int Cexplode_sepwasatend(CexplodeStrings exp_obj);

/**
 * @brief Concatenates two exp_objs into one. Modifies the first argument to contain new exp_obj.
 * Does not modify second argument 
 * @param CexplodeStrings *first pointer to CexplodeStrings type object, filled by call to Cexplode() to be combined with another CexplodeStrings object. This will contain new CexplodeStrings object holding results for both of the original CexplodeStrings objects.
 * @param CexplodeStrings *second ointer to CexplodeStrings type object, filled by call to Cexplode() to be combined with another CexplodeStrings object - this will not be modified during call.
 * @return the amount of pieces in new exp_obj - negative number upon error.
 * @warning Must not be called before calling Cexplode for both first and second argument.
 */
int Cexplode_concat(CexplodeStrings *first,CexplodeStrings *second);


/**
 * \brief removes trimchars from the beginning of a string.
 * \returns number of characters removed
 * */
int mbot_ltrim(char *text, char trimchar);

/**
 * \brief removes trailing trimchars from a string.
 * \returns number of characters removed
 * */
int mbot_rtrim(char *text, char trimchar);

/**
 * \brief removes trailing trimchars as well as trimchars from the beginning of a string.
 * \returns number of characters removed
 * */
int mbot_lrtrim(char *text, char trimchar);

/**
 * \brief removes all trimchars from a string.
 * \returns number of characters removed
 * */
int mbot_trimall(char *text, char trimchar);

/**
 * @brief Creates 32bit atomic variable, compatible with mbot_atomic* operations
 */
MbotAtomic32 * MbotAtomic32Init();
/**
 * @brief Uninitializes MbotAtomic32. This must not be called when it is possible someone is using the variable 
 * @warning If non x86 arch is used, these atomic ops are ineffective dummies using a huge semaphore (provided only for compatibility). On x86 arch compile with define ARCH_x86
 */
void MbotAtomic32Uninit(MbotAtomic32 **_this_);
/**
 * @brief Get the value atomically 
 * @warning If non x86 arch is used, these atomic ops are ineffective dummies using a huge semaphore (provided only for compatibility). On x86 arch compile with define ARCH_x86
 * */
unsigned int mbot_atomicGet(MbotAtomic32* atomic);

/**
 * @brief Increase value atomically - returns value before increment 
 * @warning If non x86 arch is used, these atomic ops are ineffective dummies using a huge semaphore (provided only for compatibility). On x86 arch compile with define ARCH_x86
 * */
unsigned int mbot_atomicAdd(MbotAtomic32* atomic,unsigned int addition);

/**
 * @brief Decrease value atomically - returns value before decrement 
 * @warning If non x86 arch is used, these atomic ops are ineffective dummies using a huge semaphore (provided only for compatibility). On x86 arch compile with define ARCH_x86
 * */
unsigned int mbot_atomicDec(MbotAtomic32* atomic,unsigned int decrement);



unsigned int mbot_atomicIncIfNequal(MbotAtomic32* atomic,unsigned int addition, unsigned int cmp);
unsigned int mbot_atomicIncIfEqual(MbotAtomic32* atomic,unsigned int decrement, unsigned int cmp);


/**
 * @brief Decrease value atomically, if original value is greater than cmp. Returns original value. (If returnval<cmp, no decrement occurred 
 * @warning If non x86 arch is used, these atomic ops are ineffective dummies using a huge semaphore (provided only for compatibility). On x86 arch compile with define ARCH_x86
 * */
unsigned int mbot_atomicDecIfGreater(MbotAtomic32* atomic,unsigned int decrement, unsigned int cmp);

/**
 * @brief Decrease value atomically, if original value is smaller than cmp. Returns original value. (If returnval>cmp, no decrement occurred 
 * @warning If non x86 arch is used, these atomic ops are ineffective dummies using a huge semaphore (provided only for compatibility). On x86 arch compile with define ARCH_x86
 * */
unsigned int mbot_atomicDecIfSmaller(MbotAtomic32* atomic,unsigned int decrement, unsigned int cmp);

/**
 * @brief Increase value atomically, if original value is greater than cmp. Returns original value. (If returnval<cmp, no increment occurred 
 * @warning If non x86 arch is used, these atomic ops are ineffective dummies using a huge semaphore (provided only for compatibility). On x86 arch compile with define ARCH_x86
 * */
unsigned int mbot_atomicIncIfGreater(MbotAtomic32* atomic,unsigned int decrement, unsigned int cmp);

/**
 * @brief Increase value atomically, if original value is smaller than cmp. Returns original value. (If returnval>cmp, no increment occurred 
 * @warning If non x86 arch is used, these atomic ops are ineffective dummies using a huge semaphore (provided only for compatibility). On x86 arch compile with define ARCH_x86
 * */
unsigned int mbot_atomicIncIfSmaller(MbotAtomic32* atomic,unsigned int decrement, unsigned int cmp);

#ifdef ARCH_x86
/**
 * @brief Performs atomic compare and swap - If original value was equal to param old, value of atomic variable is set to newval. Returns the atomic value before the operation (If returnvalue is old => swap occurred ) 
 * Should really be atomix on x86 platform, on other platforms these atomic operations are not usefull, they shall lean on semaphores ( just dummy implementations)
 * */
static __inline__ unsigned int mbot_atomicCAS(MbotAtomic32* atomic, unsigned int old, unsigned int newval)
{
    int res=old;
	__asm__ __volatile__(
    "lock  cmpxchgl %1,%2; \n\t" /* Swap value comp equals */
    : "+a" (res)    /* %0, old value to cmp - returns success */
	: "r"(newval),  /* %1, new value to set */
	  "m"(atomic->value) /* Memory address */
	: "memory");

	return res;
} 
/*
__inline__ unsigned int mbot_atomicCAS(MbotAtomic32* atomic, unsigned int old, unsigned int newval)
{
	__asm__ __volatile__(
	"lock cmpxchgl %w0,%1" 
	: "+r"(newval)
	: "m"(atomic->value), "a"(old)
	: "memory");

	return old;
} 
*/
#else
unsigned int mbot_atomicCAS(MbotAtomic32* atomic, unsigned int old, unsigned int newval);
#endif

/* Containers */

typedef struct mbot_linkedList
{
    struct mbot_linkedList *head;
    struct mbot_linkedList *next;
    struct mbot_linkedList *prev;
    void *data;
}mbot_linkedList;

/**
 * @brief Initializes linked list for use - returns ptr to list head 
 * */
mbot_linkedList *mbot_ll_init();
/**
 * @brief Gets previous list item. - returns previous item, or NULL if error occurred/first item given as param 
 * */
mbot_linkedList * mbot_ll_get_prev(mbot_linkedList *_this);
/**
 * @brief Get the head of the list 
 * Head can be used to maintain the location of empty list 
 * @return the head, and NULL on error 
 * @warning HEAD IS NOT SUPPOSED TO BE USED AS STORING ELEMENT! 
 * */
mbot_linkedList * mbot_ll_head_get(mbot_linkedList *_this);

/**
 * @brief Get's next element - NULL if error occurred, or last element was provided as argument 
 * */
mbot_linkedList * mbot_ll_get_next(mbot_linkedList *_this);

/**
 * @brief Get's the first list element - returns first element or NULL if no elements stored, or if an error occurred 
 * */
mbot_linkedList * mbot_ll_get_first(mbot_linkedList *_this);

/**
 * @brief Gets the last element in list 
 * */
mbot_linkedList * mbot_ll_get_last(mbot_linkedList *_this);

/**
 * @brief Adds item to list (data). Does not do a copy of data. Any list item (including head) can be used as _this 
 * @return list entry corresponding to stored data 
 * */
mbot_linkedList * mbot_ll_add(mbot_linkedList *_this,void *data);

/**
 * @brief removes given item from list - does not free memory. 
 * @return removed list entry, and user must call free upon entry and stored data. 
 * */
mbot_linkedList * mbot_ll_release(mbot_linkedList *_this);

/**
 * @brief removes list item which holds data pointed by data. 
 * Any list item can be given in _this. Does not free memory. Returns removed list entry, and user must call free upon entry and stored data.
 * @return removed list entry
 * */
mbot_linkedList * mbot_ll_safe_release(mbot_linkedList *_this,void *data);

/**
 * @brief Gets data stored to an entry - entry and data are left untouched 
 */
void * mbot_ll_dataGet(mbot_linkedList *_this);
/**
 * @brief Sets data to an list, 
 * @return previous data 
 * @warning - this should be avoided. Malicious use may corrupt the list! 
 * */
void * mbot_ll_dataSet(mbot_linkedList *_this,void *data);

/**
 * @brief Searchs through the list and returns element in which the held data matches data specified in params
 * @warning, all elements must contain at least as much data as specified in size_t datasize! 
 */
mbot_linkedList * mbot_ll_seek(mbot_linkedList *_this, void *data, size_t datasize);

/**
 * @brief Copies given list and itemsize bytes of data from each container to new list, and returns a pointer to the copylist 
 * @return a pointer to the copylist and NULL on error 
 * @warning This assumes that each "container" in list holds at least itemsize bytes of data - and copies exactly itemsize bytes.
 * @warning Usable really only for lists which hold fixed size items!
 */
mbot_linkedList *mbot_ll_copylist_wdata(mbot_linkedList *old,size_t itemsize);

/**
 * @brief Frees all entries from list, and destroys the list - does not free stored data. _this is NULLed upon return
 * */
void  mbot_ll_destroy(mbot_linkedList **_this);
#endif
